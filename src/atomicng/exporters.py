"""Экспорт таблицы T⁰+Δ в PLE-совместимый бинарный формат для llama.cpp-движков.

llama.cpp читает sidecar n-gram таблицы (PLE-путь, конвенция Engram) как ПОЛНУЮ таблицу
значений. Наш оверлей хранит только Δ, поэтому экспорт = слияние T⁰ источника с патчем:

    magic "DNGPLE01" | header-json len (u32 LE) | header-json | rows float32 LE (N×d, row-major)

header-json: {spec, source, seed, created_by, sources, export_version, integrity_sha256}.
Формат наш и задокументирован здесь; для конкретных форков llama.cpp адаптер добавляется
в v0.4+ по мере доступа к их кодопутям (Track vLLM/llama.cpp).
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from .exceptions import OverlayError
from .memory import MemoryPatch
from .specs import TableSpec

PLE_MAGIC = b"DNGPLE01"
PLE_EXPORT_VERSION = 1


def _materialize_full_table(base_table: np.ndarray, patch: MemoryPatch) -> np.ndarray:
    """T⁰ + Δ (материализованный) — полная таблица для движка."""
    full = np.asarray(base_table, dtype=np.float32).copy()
    if full.shape != (patch.spec.n_rows, patch.spec.row_width):
        raise OverlayError(
            f"базовая таблица {full.shape} не совпадает со спекой "
            f"({patch.spec.n_rows}, {patch.spec.row_width})"
        )
    patch.materialize()
    for r in patch.row_ids():
        full[r] += patch._rows[r]
    return full


def export_ple(base_table: np.ndarray, patch: MemoryPatch, path: str, *, source: str = "sidecar") -> str:
    """Пишет PLE-файл полной таблицы T⁰+Δ; возвращает sha256 файла."""
    full = _materialize_full_table(base_table, patch)
    header = {
        "export_version": PLE_EXPORT_VERSION,
        "spec": json.loads(patch.spec.to_json()),
        "source": source,
        "n_rows": patch.spec.n_rows,
        "row_width": patch.spec.row_width,
        "dtype": "float32-le",
        "created_by": patch.created_by,
        "sources": patch.sources,
        "n_delta_rows": len(patch),
    }
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    body = full.astype("<f4").tobytes(order="C")
    integrity = hashlib.sha256(PLE_MAGIC + header_bytes + body).hexdigest()
    header["integrity_sha256"] = integrity
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    with open(path, "wb") as f:
        f.write(PLE_MAGIC)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(body)
    return integrity


def load_ple(path: str) -> tuple[TableSpec, np.ndarray, dict]:
    """Читает PLE-файл → (спека, полная таблица, header). Строгая валидация."""
    data = Path(path).read_bytes()
    if data[: len(PLE_MAGIC)] != PLE_MAGIC:
        raise OverlayError(f"{path!r}: не PLE-файл (magic не совпал)")
    (header_len,) = struct.unpack_from("<I", data, len(PLE_MAGIC))
    off = len(PLE_MAGIC) + 4
    header = json.loads(data[off : off + header_len].decode("utf-8"))
    body = data[off + header_len :]
    expected = header.get("integrity_sha256")
    if expected is not None:
        no_integrity = {k: v for k, v in header.items() if k != "integrity_sha256"}
        header_bytes = json.dumps(no_integrity, ensure_ascii=False).encode("utf-8")
        actual = hashlib.sha256(PLE_MAGIC + header_bytes + body).hexdigest()
        if actual != expected:
            raise OverlayError(f"{path!r}: манифест целостности PLE не совпал (файл повреждён)")
    spec = TableSpec.from_json(json.dumps(header["spec"]))
    table = np.frombuffer(body, dtype="<f4").reshape(header["n_rows"], header["row_width"]).copy()
    return spec, table, header
