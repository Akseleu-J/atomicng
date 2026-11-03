"""Проверка неразрушаемости (P0/S3): очистка Δ возвращает базовое поведение, restore точен.

Протокол: нейтральные последовательности генерируются ОДИН РАЗ (не адресующие строки
патча — на плотной sidecar-таблице случайные пробы задевают записанные строки, и их
сдвиг легитимен), затем сравниваются выходы до/после очистки на одних и тех же входах.
Точность restore проверяется чексуммой патча — модель-независимая побитовая проверка.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class VerifyReport:
    passed: bool
    n_probes: int
    max_abs_diff: float
    message: str


def _make_probe(model, n_probes: int) -> Callable[[int], np.ndarray]:
    """Единый набор входов до и после очистки: neutral_token_seqs или probe как fallback."""
    seqs = None
    if hasattr(model, "neutral_token_seqs"):
        seqs = model.neutral_token_seqs(n_probes)
        return lambda _k: model.run_seqs(seqs)
    return lambda k: model.probe(k)


def verify_deletion(model, n_probes: int = 8) -> VerifyReport:
    probe = _make_probe(model, n_probes)
    before = probe(n_probes)
    checksum_before = model.memory.checksum()
    snap = model.memory.snapshot()
    model.memory.clear()
    after = probe(n_probes)
    model.memory.restore(snap)
    checksum_restored = model.memory.checksum()

    diff = float(np.max(np.abs(before - after))) if before.size else 0.0
    probes_ok = bool(np.array_equal(before, after))
    restore_ok = checksum_restored == checksum_before
    if probes_ok and restore_ok:
        message = "откат побитовый (пробы нейтральны, restore точен)"
    elif not probes_ok:
        message = "ОШИБКА: очистка Δ изменила выходы на нейтральных пробах"
    else:
        message = "ОШИБКА: restore не восстановил патч байт-в-байт"
    return VerifyReport(
        passed=probes_ok and restore_ok,
        n_probes=n_probes,
        max_abs_diff=diff,
        message=message,
    )
