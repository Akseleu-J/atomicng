"""Разреженный патч Δ поверх замороженной таблицы T⁰:  T = T⁰ + Δ.

Инварианты (P0, S2):
- T⁰ в MemoryPatch не хранится и недоступна для записи — патч знает только Δ;
- хранятся только затронутые строки, счётчики записей и ленивый decay (block clock);
- очистка патча структурно возвращает исходную модель;
- .atomicng-оверлей — data-only артефакт с sha256-манифестом целостности (S1).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator

import numpy as np

from .exceptions import OverlayError
from .specs import SPEC_VERSION, TableSpec

OVERLAY_FORMAT = "atomicng-overlay"
OVERLAY_VERSION = 2


def _arrays_checksum(row_ids: np.ndarray, deltas: np.ndarray, counts: np.ndarray, clock: int) -> str:
    h = hashlib.sha256()
    h.update(row_ids.astype("<i8").tobytes())
    h.update(deltas.astype("<f4").tobytes())
    h.update(counts.astype("<i8").tobytes())
    h.update(int(clock).to_bytes(8, "little"))
    return h.hexdigest()


class MemoryPatch:
    def __init__(self, spec: TableSpec, decay: float = 1.0, *, created_by: str = "") -> None:
        if not 0.0 < decay <= 1.0:
            raise OverlayError(f"decay={decay} вне (0, 1]")
        self.spec = spec
        self.decay = float(decay)
        self.created_by = created_by
        self._rows: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}
        self._last_touch: dict[int, int] = {}
        self._clock = 0
        self.sources: list[str] = []  # provenance: идентификаторы записанных документов/партий

    # ------------------------------------------------------------------ чтение

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, row_id: int) -> bool:
        return row_id in self._rows

    def row_ids(self) -> Iterator[int]:
        return iter(self._rows)

    def get(self, row_id: int) -> np.ndarray:
        """Эффективное значение Δ_j с учётом ленивого decay на текущем такте."""
        row = self._rows.get(row_id)
        if row is None:
            return np.zeros(self.spec.row_width, dtype=np.float32)
        factor = self.decay ** (self._clock - self._last_touch[row_id])
        if factor == 1.0:
            return row.copy()
        return (factor * row).astype(np.float32)

    def count(self, row_id: int) -> int:
        return self._counts.get(row_id, 0)

    @property
    def total_writes(self) -> int:
        return sum(self._counts.values())

    # ---------------------------------------------------------- запись (rules)

    def bump_clock(self) -> None:
        """Новый блок записи: такт decay сдвигается на единицу."""
        self._clock += 1

    def _write_row(self, row_id: int, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (self.spec.row_width,):
            raise OverlayError(f"ожидалась строка ширины {self.spec.row_width}, получено {arr.shape}")
        self._rows[row_id] = arr
        self._last_touch[row_id] = self._clock

    def _add_count(self, row_id: int, n: int) -> None:
        self._counts[row_id] = self._counts.get(row_id, 0) + n

    def _set_row_state(
        self, row_id: int, value: np.ndarray | None, count: int, last_touch: int
    ) -> None:
        """Прямое восстановление состояния строки (undo/restore). value=None → строка удаляется."""
        if value is None:
            self._rows.pop(row_id, None)
            self._counts.pop(row_id, None)
            self._last_touch.pop(row_id, None)
            return
        self._rows[row_id] = np.asarray(value, dtype=np.float32).copy()
        self._counts[row_id] = count
        self._last_touch[row_id] = last_touch

    # ------------------------------------------------------------ undo / verify

    def preimage(self, row_id: int) -> tuple[np.ndarray | None, int, int]:
        """Состояние строки ДО записи блока: (значение|None, счётчик, last_touch)."""
        if row_id not in self._rows:
            return None, 0, self._clock
        return self.get(row_id), self._counts[row_id], self._last_touch[row_id]

    def snapshot(self) -> dict:
        return {
            "clock": self._clock,
            "rows": {r: v.copy() for r, v in self._rows.items()},
            "counts": dict(self._counts),
            "last_touch": dict(self._last_touch),
        }

    def restore(self, snap: dict) -> None:
        self.clear()
        self._clock = snap["clock"]
        for r, v in snap["rows"].items():
            self._rows[r] = v.copy()
        self._counts.update(snap["counts"])
        self._last_touch.update(snap["last_touch"])

    def clear(self) -> None:
        """Kill-switch: Δ=0, модель структурно равна базовой."""
        self._rows.clear()
        self._counts.clear()
        self._last_touch.clear()
        self._clock = 0

    # ----------------------------------------------------------------- merge

    def merge(self, other: MemoryPatch, gate: float = 0.5) -> dict:
        """Слияние оверлеев по D1-политике: строки other — одиночные события с гейтом gate.

        Возвращает отчёт. Спеки должны совпадать; пустой other — no-op.
        """
        from .rules.d1 import D1Rule, Event  # локальный импорт: rules зависят от memory

        if other.spec != self.spec:
            raise OverlayError(f"слияние: спека {other.spec.name!r} не совпадает с {self.spec.name!r}")
        self.materialize()
        other.materialize()
        events = [Event(row_id=r, target=other._rows[r], gate=gate) for r in sorted(other._rows)]
        report = D1Rule().apply(self, events)
        self.sources.append(f"merge@clock{self._clock}:{len(other._rows)}rows")
        return report.to_dict()

    # ----------------------------------------------------------------- отчёт

    def stats(self) -> dict:
        norms = {r: float(np.linalg.norm(v)) for r, v in self._rows.items()}
        reused = sum(1 for c in self._counts.values() if c > 1)
        return {
            "n_rows": len(self._rows),
            "total_writes": self.total_writes,
            "reused_rows": reused,
            "reuse_ratio": reused / len(self._rows) if self._rows else 0.0,
            "clock": self._clock,
            "max_norm": max(norms.values(), default=0.0),
            "mean_norm": float(np.mean(list(norms.values()))) if norms else 0.0,
            "top_rows": sorted(norms, key=norms.get, reverse=True)[:8],
            "sources": len(self.sources),
        }

    def checksum(self) -> str:
        """sha256 состояния патча (для audit-лога, verify и сравнения состояний)."""
        rows = sorted(self._rows)
        ids = np.asarray(rows, dtype=np.int64)
        deltas = np.stack([self._rows[r] for r in rows]) if rows else np.zeros((0, self.spec.row_width), np.float32)
        counts = np.asarray([self._counts[r] for r in rows], dtype=np.int64)
        return _arrays_checksum(ids, deltas, counts, self._clock)

    # ------------------------------------------------------------- оверлей .atomicng

    def materialize(self) -> None:
        """Применяет ленивый decay к хранимым значениям (перед сохранением/слиянием)."""
        for r in self._rows:
            self._rows[r] = self.get(r)
            self._last_touch[r] = self._clock

    def save_pretrained(self, path: str) -> None:
        """Сохраняет .atomicng-оверлей: npz + json-метаданные + sha256-манифест (data-only)."""
        self.materialize()
        rows = sorted(self._rows)
        ids = np.asarray(rows, dtype=np.int64)
        deltas = (
            np.stack([self._rows[r] for r in rows])
            if rows
            else np.zeros((0, self.spec.row_width), np.float32)
        )
        counts = np.asarray([self._counts[r] for r in rows], dtype=np.int64)
        meta = {
            "overlay_format": OVERLAY_FORMAT,
            "overlay_version": OVERLAY_VERSION,
            "core_spec_version": SPEC_VERSION,
            "spec": json.loads(self.spec.to_json()),
            "decay": self.decay,
            "clock": self._clock,
            "created_by": self.created_by,
            "sources": self.sources,
            "integrity": _arrays_checksum(ids, deltas, counts, self._clock),
        }
        np.savez(
            path,
            row_ids=ids,
            deltas=deltas,
            counts=counts,
            meta=np.asarray(json.dumps(meta)),
        )

    @classmethod
    def load_pretrained(cls, path: str, spec: TableSpec | None = None) -> MemoryPatch:
        """Строгая загрузка (S1): неизвестная версия или несовпадение sha256 → отказ."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("overlay_format") != OVERLAY_FORMAT or meta.get("overlay_version") != OVERLAY_VERSION:
                raise OverlayError(
                    f"оверлей {path!r}: формат {meta.get('overlay_format')!r} "
                    f"v{meta.get('overlay_version')} не поддерживается"
                )
            ids = np.asarray(data["row_ids"], dtype=np.int64)
            deltas = np.asarray(data["deltas"], dtype=np.float32)
            counts = np.asarray(data["counts"], dtype=np.int64)
            clock = int(meta["clock"])
            expected = meta.get("integrity")
            if expected is not None and _arrays_checksum(ids, deltas, counts, clock) != expected:
                raise OverlayError(f"оверлей {path!r}: манифест целостности не совпал (файл повреждён)")

            loaded_spec = TableSpec.from_json(json.dumps(meta["spec"]))
            if spec is not None and loaded_spec != spec:
                raise OverlayError(
                    f"спека оверлея {loaded_spec.name!r} не совпадает с ожидаемой {spec.name!r}"
                )
            if deltas.shape[1] != loaded_spec.row_width:
                raise OverlayError(
                    f"оверлей {path!r}: ширина строки {deltas.shape[1]} != {loaded_spec.row_width}"
                )
            patch = cls(loaded_spec, decay=float(meta["decay"]), created_by=meta.get("created_by", ""))
            patch._clock = clock
            patch.sources = list(meta.get("sources", []))
            for i, r in enumerate(ids):
                patch._rows[int(r)] = deltas[i].copy()
                patch._counts[int(r)] = int(counts[i])
                patch._last_touch[int(r)] = clock
        return patch
