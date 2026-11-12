"""Versioned n-gram table specifications for the NumPy core.

A specification records the row count, row width, and hash key scheme.
It is included in each overlay artifact.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

SPEC_FORMAT = "atomicng-spec"
SPEC_VERSION = 1


@dataclass(frozen=True)
class TableSpec:
    """Geometry of Table T.

    n_rows       — number of lines (the sidecar-prototype article 16 384));
    row_width    — width of line d (in article 160));
    n_heads      — number of hash heads (in the native table 16:8 bigrams) + 8 trigrams;
                   sidecar v0.1 - single-headed);
    key_order    — How many of the last tokens is the key built (2 = bigram, 3 = trigram));
    inject_layer — before the attention of which layer the table is read (in the article - before the 2nd)).
    """

    name: str
    n_rows: int = 16_384
    row_width: int = 160
    n_heads: int = 1
    key_order: int = 2
    inject_layer: int = 2

    def __post_init__(self) -> None:
        if self.n_rows <= 0 or self.row_width <= 0:
            raise ValueError("n_rows and row_width must be positive.")
        if self.key_order not in (2, 3):
            raise ValueError("key_order must be 2 (bigram) or 3 (trigram).")
        if self.n_heads < 1:
            raise ValueError("n_heads >= 1")

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "spec_format": SPEC_FORMAT, "spec_version": SPEC_VERSION})

    @classmethod
    def from_json(cls, raw: str) -> TableSpec:
        data = json.loads(raw)
        version = data.pop("spec_version", None)
        data.pop("spec_format", None)
        if version != SPEC_VERSION:
            raise ValueError(f"Unsupported spec_version {version!r}. Expected {SPEC_VERSION}.")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


SIDECAR_SMALL = TableSpec(name="sidecar-small")
"""Sidecar-prototype table from the article: 16,384×160, external to any HF-model."""
