"""Адресация n-грамм: два схемы хеша.

- blake2b — точная совместимость с NumPy-ядром (TinyModel.row_hash) и HashWriter;
  считается на CPU построчно (медленный, но канонический путь);
- poly64   — векторизованный GPU-путь: полиномиальный катящийся хеш по окну из
  key_order токенов со splitmix-финализатором на int64 (переполнение — обычное
  для хеша заворачивание двух complement, детерминировано на CPU и CUDA).

Субтаблица фиксирует схему в своей спецификации: запись и чтение обязаны
использовать одну и ту же функцию (инвариант проверяется тестами).
"""
from __future__ import annotations

import hashlib

import torch

HASH_SCHEMES = ("blake2b", "poly64")

# Целые константы splitmix64 (беззнаковые в каноне; в int64 используем младшие 64 бита).
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_PRIME = 0x100000001B3  # FNV-простое для катящегося полинома


def _blake_digest(key: str) -> int:
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def blake_addresses(tokens: list[int], n_rows: int, key_order: int = 2) -> list[int]:
    """Адреса окон длины key_order: out[w] = hash(tokens[w : w+key_order]).

    Для key_order=2 совпадает с TinyModel.row_hash(tokens[w], tokens[w+1]).
    """
    if len(tokens) < key_order:
        return []
    return [_blake_digest(":".join(str(int(t)) for t in tokens[w : w + key_order])) % n_rows for w in range(len(tokens) - key_order + 1)]


def _wrap64(x: int) -> int:
    """Python-int → знаковый int64 (модульная арифметика кольца 2⁶⁴)."""
    x %= 1 << 64
    return x - (1 << 64) if x >= (1 << 63) else x


def poly_addresses(ids: torch.Tensor, n_rows: int, key_order: int = 2) -> torch.Tensor:
    """(B, T) int64 → (B, T-key_order+1) int64 адреса на том же устройстве.

    out[w] = poly(ids[w : w+key_order]) mod n_rows; выравнивание совпадает с
    TorchTinyModel.bigram_rows для key_order=2 (строка читается на позиции w+1).
    Окно катится по модулю 2⁶⁴: вычитание старшего токена и умножение на
    нечётное простое P обратимы в кольце, поэтому конгруэнтность сохраняется.
    """
    if ids.dtype != torch.int64:
        ids = ids.long()
    if ids.shape[-1] < key_order:
        raise ValueError(f"Sequence length {ids.shape[-1]} < key_order {key_order}.")
    lead = _wrap64(_PRIME ** (key_order - 1))  # вес старшего токена окна
    h = ids[..., 0] * lead
    for i in range(1, key_order):
        h = h * _PRIME + ids[..., i]
    cols = [h]
    for w in range(1, ids.shape[-1] - key_order + 1):
        h = (h - ids[..., w - 1] * lead) * _PRIME + ids[..., w + key_order - 1]
        cols.append(h)
    h = torch.stack(cols, dim=-1)
    # splitmix64-финализатор на int64 (заворачивающее умножение, сдвиги вправо).
    h = h ^ (h >> 30)
    h = h * _MIX1
    h = h ^ (h >> 27)
    h = h * _MIX2
    h = h ^ (h >> 31)
    return torch.remainder(h, n_rows)  # знак остатка — от делителя → адрес ≥ 0


def addresses(
    input_ids: torch.Tensor,
    n_rows: int,
    key_order: int = 2,
    scheme: str = "poly64",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Единая точка входа: (B, T) → (B, T-key_order+1) адреса на устройстве.

    blake2b считается на CPU (точность совместимости с ядром), poly64 — прямо
    на устройстве входа.
    """
    if scheme not in HASH_SCHEMES:
        raise ValueError(f"Unknown hash scheme {scheme!r}. Available: {HASH_SCHEMES}.")
    if scheme == "poly64":
        return poly_addresses(input_ids.to(device or input_ids.device), n_rows, key_order)
    rows: list[list[int]] = []
    for seq in input_ids.detach().cpu().tolist():
        rows.append(blake_addresses(seq, n_rows, key_order))
    target = torch.device(device) if device is not None else input_ids.device
    if not rows:
        return torch.zeros((input_ids.shape[0], 0), dtype=torch.int64, device=target)
    return torch.as_tensor(rows, dtype=torch.int64, device=target)
