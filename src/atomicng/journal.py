"""Append-only JSONL audit logging and exact pre-image undo.

Before writing a block, save the affected rows, their counters, and the clock.
Reverting restores the saved state byte for byte.

Undo semantics (author review, issue #1): undo is LAST-IN, FIRST-OUT. `revert_last`
is always exact for the most recent block. Reverting an OLDER block after later
blocks have touched the same rows is not exact — D1 and the gate depend on the
current row state, so the pre-image journal only guarantees exact rollback in
LIFO order. For non-LIFO scenarios use overlay merge/revert semantics instead.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .memory import MemoryPatch


class AuditLog:
    """Append-only log of operations. write/merge/clear leaves a note with a report."""

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path) if path else None
        self.entries: list[dict] = []
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                self.entries = [json.loads(line) for line in f if line.strip()]

    def append(self, kind: str, payload: dict) -> None:
        entry = {"kind": kind, "payload": payload}
        self.entries.append(entry)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.entries)


@dataclass
class _UndoEntry:
    clock: int
    rows: dict[int, tuple[np.ndarray | None, int, int]] = field(default_factory=dict)


class UndoJournal:
    """Store pre-images so revert_last can restore the patch before the last write."""

    def __init__(self) -> None:
        self._stack: list[_UndoEntry] = []

    def record(self, patch: MemoryPatch, row_ids: Iterable[int]) -> None:
        entry = _UndoEntry(clock=patch._clock)
        for r in set(row_ids):
            value, count, last_touch = patch.preimage(r)
            entry.rows[r] = (value, count, last_touch)
        self._stack.append(entry)

    def revert_last(self, patch: MemoryPatch) -> list[int]:
        if not self._stack:
            return []
        entry = self._stack.pop()
        for r, (value, count, last_touch) in entry.rows.items():
            patch._set_row_state(r, value, count, last_touch)
        patch._clock = entry.clock
        return sorted(entry.rows)

    def __len__(self) -> int:
        return len(self._stack)
