"""L2Subtable — плотный «том» памяти: значения (n_rows, d) + маска занятости.

Собирается из материалезованного MemoryPatch (наполненного тем же D1) и живёт
на устройстве как единый тензор: gather адреса → готовые вектора без тяжёлых
вычислений. На диске хранится квантованно (int8 + глобальный масштаб) с
sha256-манифестом целостности — конвенции overlay из memory.py.

Схема хеша адресов входит в спецификацию: запись и чтение субтаблицы обязаны
использовать одну схему (blake2b — совместимость с ядром, poly64 — GPU-горячий
путь).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..exceptions import OverlayError
from ..memory import MemoryPatch
from ..specs import TableSpec
from .hashing import HASH_SCHEMES

SUBTABLE_FORMAT = "atomicng-hngram-subtable"
SUBTABLE_VERSION = 1


@dataclass(frozen=True)
class SubtableSpec:
    """Геометрия и схема адресации одного L2-тома."""

    name: str
    n_rows: int = 16_384
    row_width: int = 160
    key_order: int = 2
    hash_scheme: str = "poly64"

    def __post_init__(self) -> None:
        if self.n_rows <= 0 or self.row_width <= 0:
            raise ValueError("n_rows and row_width must be positive.")
        if self.key_order not in (2, 3):
            raise ValueError("key_order must be 2 (bigram) or 3 (trigram).")
        if self.hash_scheme not in HASH_SCHEMES:
            raise ValueError(f"hash_scheme must be one of {HASH_SCHEMES}.")

    @classmethod
    def from_table_spec(cls, spec: TableSpec, name: str | None = None, hash_scheme: str = "blake2b") -> SubtableSpec:
        """Субтаблица, адресно-совместимая с NumPy-ядром (blake2b)."""
        return cls(
            name=name or f"{spec.name}-l2",
            n_rows=spec.n_rows,
            row_width=spec.row_width,
            key_order=spec.key_order,
            hash_scheme=hash_scheme,
        )

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "subtable_format": SUBTABLE_FORMAT, "subtable_version": SUBTABLE_VERSION})

    @classmethod
    def from_json(cls, raw: str) -> SubtableSpec:
        data = json.loads(raw)
        version = data.pop("subtable_version", None)
        data.pop("subtable_format", None)
        if version != SUBTABLE_VERSION:
            raise ValueError(f"Unsupported subtable_version {version!r}. Expected {SUBTABLE_VERSION}.")
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def _subtable_checksum(values_i8: np.ndarray, occupancy: np.ndarray, scale: np.float32) -> str:
    h = hashlib.sha256()
    h.update(values_i8.astype("<i1").tobytes())
    h.update(occupancy.astype(bool).tobytes())
    h.update(np.asarray(scale, dtype="<f4").tobytes())
    return h.hexdigest()


class L2Subtable:
    """Плотный L2-том на устройстве. Значения — Δ-вектора (дельты к ядру)."""

    def __init__(self, spec: SubtableSpec, values: torch.Tensor, occupancy: torch.Tensor) -> None:
        if values.shape != (spec.n_rows, spec.row_width):
            raise OverlayError(f"Expected values ({spec.n_rows}, {spec.row_width}), got {tuple(values.shape)}.")
        if occupancy.shape != (spec.n_rows,):
            raise OverlayError(f"Expected occupancy ({spec.n_rows},), got {tuple(occupancy.shape)}.")
        self.spec = spec
        self.values = values.float()
        self.occupancy = occupancy.bool()

    # ------------------------------------------------------------- сборка

    @classmethod
    def from_patch(cls, patch: MemoryPatch, spec: SubtableSpec | None = None) -> L2Subtable:
        """Материализация разреженного патча (наполненного D1) в плотный том.

        Адресация патча остаётся как была (row_id из HashWriter → blake2b),
        поэтому по умолчанию схема наследуется как blake2b.
        """
        spec = spec or SubtableSpec.from_table_spec(patch.spec)
        if spec.n_rows < patch.spec.n_rows or spec.row_width != patch.spec.row_width:
            raise OverlayError(f"Patch spec {patch.spec.name!r} does not fit subtable spec {spec.name!r}.")
        values = torch.zeros((spec.n_rows, spec.row_width), dtype=torch.float32)
        occupancy = torch.zeros(spec.n_rows, dtype=torch.bool)
        for row_id, vec in patch._rows.items():
            if row_id >= spec.n_rows:
                continue
            values[int(row_id)] = torch.as_tensor(np.asarray(vec, dtype=np.float32))
            occupancy[int(row_id)] = True
        return cls(spec, values, occupancy)

    # ------------------------------------------------------------- чтение

    def lookup(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(…,) адреса → (значения (…, d), hits (…,)) на устройстве тома.

        Промах (строка не занята) даёт нулевой вектор — деградация до чистого L1
        без ошибок: правило отката концепции H-Ngram.
        """
        if rows.numel() and (rows.min() < 0 or rows.max() >= self.spec.n_rows):
            raise ValueError("Address out of range for subtable.")
        vals = self.values[rows.long()]
        hits = self.occupancy[rows.long()]
        return vals * hits[..., None].to(vals.dtype), hits

    def hit_rate(self, rows: torch.Tensor) -> float:
        """Доля занятых строк в пробе адресов (метрика покрытия тома)."""
        if rows.numel() == 0:
            return 0.0
        return float(self.occupancy[rows.long()].float().mean())

    # ----------------------------------------------------------- устройство

    def to(self, device: torch.device | str) -> L2Subtable:
        self.values = self.values.to(device)
        self.occupancy = self.occupancy.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.values.device

    # ------------------------------------------------------------ диск

    def save_pretrained(self, path: str) -> None:
        """Квантованное сохранение: int8 + глобальный масштаб + sha256-манифест."""
        values = self.values.detach().cpu().numpy().astype(np.float32)
        scale = np.float32(max(np.abs(values).max() / 127.0, 1e-12)) if values.size else np.float32(1.0)
        values_i8 = np.rint(values / scale).astype(np.int8)
        occupancy = self.occupancy.detach().cpu().numpy().astype(bool)
        meta = {
            "spec": json.loads(self.spec.to_json()),
            "integrity": _subtable_checksum(values_i8, occupancy, scale),
        }
        np.savez(path, values_i8=values_i8, occupancy=occupancy, scale=np.asarray(scale, dtype=np.float32),
                 meta=np.asarray(json.dumps(meta)))

    @classmethod
    def load_pretrained(cls, path: str, device: torch.device | str = "cpu") -> L2Subtable:
        """Строгая загрузка: неизвестная версия или сбой sha256 → отказ."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            spec = SubtableSpec.from_json(json.dumps(meta["spec"]))
            values_i8 = np.asarray(data["values_i8"], dtype=np.int8)
            occupancy = np.asarray(data["occupancy"], dtype=bool)
            scale = np.asarray(data["scale"], dtype=np.float32).item()
            expected = meta.get("integrity")
            if expected is not None and _subtable_checksum(values_i8, occupancy, np.float32(scale)) != expected:
                raise OverlayError(f"subtable {path!r}: integrity check failed")
        values = torch.as_tensor(values_i8, dtype=torch.float32) * float(scale)
        return cls(spec, values, torch.as_tensor(occupancy)).to(device)

    # ----------------------------------------------------------- отчёт

    def stats(self) -> dict:
        norms = self.values.norm(dim=1)
        return {
            "name": self.spec.name,
            "n_rows": self.spec.n_rows,
            "row_width": self.spec.row_width,
            "key_order": self.spec.key_order,
            "hash_scheme": self.spec.hash_scheme,
            "fill_ratio": float(self.occupancy.float().mean()),
            "mean_norm": float(norms[self.occupancy].mean()) if self.occupancy.any() else 0.0,
            "max_norm": float(norms.max()) if self.occupancy.any() else 0.0,
        }
