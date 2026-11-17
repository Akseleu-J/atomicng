"""Deterministic hash writer for smoke tests.

Text bigrams address rows, and normalized token embeddings provide targets
with a fixed gate. This exercises the write, read, clear, and verification path.
"""
from __future__ import annotations

import numpy as np

from .backends.tiny import TinyModel
from .rules.d1 import Event
from .specs import TableSpec

TARGET_SCALE = 0.05


class HashWriter:
    def __init__(self, spec: TableSpec | None = None, vocab_size: int = 512, gate: float = 0.5) -> None:
        self.spec = spec or TableSpec(name="hash-writer")
        self.vocab_size = vocab_size
        self.gate = float(gate)
        self._targets = np.random.default_rng(17).standard_normal((vocab_size, self.spec.row_width))

    def tokenize(self, text: str) -> list[int]:
        """Deterministic tokenization without dependencies: UTF-8 bytes → id."""
        return [b % self.vocab_size for b in text.encode("utf-8")]

    def events(self, texts: list[str]) -> list[Event]:
        events: list[Event] = []
        for text in texts:
            tokens = self.tokenize(text)
            for i in range(1, len(tokens)):
                row = TinyModel.row_hash(tokens[i - 1], tokens[i], self.spec.n_rows)
                target = TARGET_SCALE * self._targets[tokens[i]] / np.linalg.norm(self._targets[tokens[i]])
                events.append(Event(row_id=row, target=target.astype(np.float32), gate=self.gate))
        return events
