"""Hit-последовательности для negative-control (по запросу u/amargolin на r/LocalLLM).

covering-set: для КАЖДОЙ строки патча генерируется проба, чья биграмма бьёт точно в эту
строку — детерминированный перебор пар (гарантия нахождения). Только covering-set ловит
неполный clear: нейтральные пробы слепы к патчу, а случайные hit-пробы могут не накрыть
оставшиеся после бага строки.
"""
from __future__ import annotations

import hashlib

from .backends.tiny import TinyModel  # noqa: TC001 — тип в сигнатуре


def hit_sequences(model: TinyModel, n: int = 8, seq_len: int = 12, seed: int = 99) -> list[list[int]]:
    """Пробы, покрывающие все строки патча: по одной на строку (cap 512 строк)."""
    written = sorted(model.memory.row_ids()) if model.memory is not None else []
    if not written:
        return []
    V = model.vocab_size
    N = model.spec.n_rows
    probes: list[list[int]] = []
    for r in written[:512]:
        a, b = _pair_for_row(r, V, N)
        tail = [(a + b + i) % V for i in range(max(seq_len - 2, 0))]
        probes.append([a, b] + tail)
    return probes


def _pair_for_row(r: int, V: int, N: int) -> tuple[int, int]:
    """Детерминированный поиск токен-пары, чей биграммный хэш = r."""
    for c in range(V * V):
        a, b = c % V, (c // V) % V
        digest = hashlib.blake2b(f"{a}:{b}".encode(), digest_size=8).digest()
        if int.from_bytes(digest, "little") % N == r:
            return a, b
    raise RuntimeError(f"hit_sequences: пара для строки {r} не найдена за {V * V} попыток")


def checksum_detects_in_memory_tamper(model: TinyModel, row_id: int) -> bool:
    """Манипуляция значением строки патча ловится чексуммой (не probe-verify)."""
    import numpy as np

    before = model.memory.checksum()
    original = model.memory._rows[row_id].copy()
    model.memory._rows[row_id] = original + np.float32(1.0)
    after = model.memory.checksum()
    model.memory._rows[row_id] = original
    return after != before
