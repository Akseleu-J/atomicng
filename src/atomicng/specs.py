"""Спецификации n-gram таблиц (слой Core, без ML-фреймворков).

Спека описывает таблицу T: число строк, ширину строки, схему хэш-ключей.
Спека версионируется и сохраняется внутрь .atomicng-оверлея (S1 из плана).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

SPEC_FORMAT = "atomicng-spec"
SPEC_VERSION = 1


@dataclass(frozen=True)
class TableSpec:
    """Геометрия таблицы T.

    n_rows       — число строк (у sidecar-прототипа статьи 16 384);
    row_width    — ширина строки d (в статье 160);
    n_heads      — число хэш-голов (в нативной таблице 16: 8 биграммных + 8 триграммных;
                   sidecar v0.1 — одно-головая);
    key_order    — из скольких последних токенов строится ключ (2 = биграмма, 3 = триграмма);
    inject_layer — перед attention какого слоя читается таблица (в статье — перед 2-м).
    """

    name: str
    n_rows: int = 16_384
    row_width: int = 160
    n_heads: int = 1
    key_order: int = 2
    inject_layer: int = 2

    def __post_init__(self) -> None:
        if self.n_rows <= 0 or self.row_width <= 0:
            raise ValueError("n_rows и row_width должны быть положительными")
        if self.key_order not in (2, 3):
            raise ValueError("key_order поддерживает 2 (биграмма) и 3 (триграмма)")
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
            raise ValueError(f"spec_version {version!r} не поддерживается (ожидалась {SPEC_VERSION})")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


SIDECAR_SMALL = TableSpec(name="sidecar-small")
"""Sidecar-таблица прототипа из статьи: 16 384 × 160, внешняя к любой HF-модели."""
