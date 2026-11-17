"""Детерминированный наполнитель L2-субтаблицы (HashWriter-стиль, D1-семантика).

Строка биграммы (t_{i-1}, t_i) получает цель = масштабированный нормированный
эмбеддинг ПРОДОЛЖЕНИЯ t_{i+1}: delta-смысл строки — «что обычно следует после
этой связки». Чтение на позиции j добавляет в резидуальный поток вектор,
сонаправленный эмбеддингу следующего токена → логиты продолжения растут.

Это черновой путь без обучения: writer-MLP (writers.mlp.WriterMLP + трейнер)
подключается на то же место — заменой цели события. Семантика записи одна:
блок-усреднение D1 по строкам (rules.d1).
"""
from __future__ import annotations

import numpy as np
import torch

from ..memory import MemoryPatch
from ..rules.d1 import D1Rule, Event
from ..specs import TableSpec
from .hashing import addresses
from .store import L2Subtable, SubtableSpec

TARGET_SCALE = 0.05


def _continuation_targets(vocab_size: int, row_width: int, seed: int = 17) -> np.ndarray:
    """Сеединный словарь (vocab, d): эмбеддинги для целей событий."""
    rng = np.random.default_rng(seed)
    return (0.10 * rng.standard_normal((vocab_size, row_width))).astype(np.float32)


def fill_subtable_from_sequences(
    seqs: list[list[int]],
    spec: SubtableSpec,
    vocab_size: int,
    *,
    gate: float = 0.5,
    target_scale: float = TARGET_SCALE,
    seed: int = 17,
) -> L2Subtable:
    """Наполнить L2-субтаблицу из корпусных последовательностей.

    Каждый триплет (t_{i-1}, t_i, t_{i+1}) даёт событие: строка hash(t_{i-1}, t_i)
    ← target = target_scale · ē(t_{i+1}). Один вызов apply = один блок D1
    (порядок событий внутри блока не влияет на результат — инвариант D1).
    """
    if gate < 0.0 or gate > 1.0:
        raise ValueError("gate must be in [0, 1].")
    table_spec = TableSpec(
        name=spec.name, n_rows=spec.n_rows, row_width=spec.row_width,
        n_heads=1, key_order=spec.key_order, inject_layer=2,
    )
    patch = MemoryPatch(table_spec)
    targets = _continuation_targets(vocab_size, spec.row_width, seed)

    events: list[Event] = []
    for seq in seqs:
        if len(seq) < spec.key_order + 1:
            continue
        rows = _addresses_for_seq(seq, spec, vocab_size)
        # rows[w] = адрес окна seq[w : w+k] → читается на позиции w+k-1;
        # продолжение после окна — seq[w+k], т.е. target[rows[w]] = ē(seq[w+k]).
        # Последнее окно пропускается: внутри последовательности нет продолжения.
        for w in range(len(rows) - 1):
            row = rows[w]
            nxt = seq[w + spec.key_order]
            vec = targets[nxt]
            vec = vec / max(float(np.linalg.norm(vec)), 1e-8)
            events.append(Event(row_id=int(row), target=(target_scale * vec).astype(np.float32), gate=gate))
    D1Rule().apply(patch, events)
    return L2Subtable.from_patch(patch, spec)


def _addresses_for_seq(seq: list[int], spec: SubtableSpec, vocab_size: int) -> list[int]:
    """Адреса окон по схеме субтаблицы (инвариант запись==чтение).

    blake2b-путь каноничен: совпадает с TinyModel.row_hash для key_order=2.
    """
    from .hashing import blake_addresses

    if spec.hash_scheme == "blake2b":
        return blake_addresses(seq, spec.n_rows, spec.key_order)
    ids = torch.as_tensor([seq], dtype=torch.int64)
    rows = addresses(ids, spec.n_rows, spec.key_order, spec.hash_scheme, "cpu")
    return [int(r) for r in rows[0]]
