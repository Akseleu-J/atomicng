"""Negative control для verify_deletion (запрос u/amargolin на r/LocalLLM).

Контракт проверки:
- нейтральные пробы (не бьют в патч) до/после clear совпадают побитово —
  локальность Δ на несвязанном тексте;
- hit-пробы (бьют в строки патча) ПОСЛЕ clear совпадают с эталонной моделью
  (та же архитектура, seed, пустая память) — ловит неполный clear;
- манипуляция значением строки патча ловится чексуммой, не probe-verify.

Negative control: эмулируем «баг» clear (удаляет только половину затронутых строк) —
self-comparison это пропускает, reference-сравнение падает. Проверка умеет падать.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import atomicng
from atomicng import TinyModel
from atomicng.hit_probes import checksum_detects_in_memory_tamper, hit_sequences
from atomicng.verify import verify_deletion


def _model_with_docs():
    model = TinyModel()
    writer = atomicng.HashWriter(model.spec)
    for i in range(4):
        atomicng.write(model, f"negative control document {i} with facts and numbers {i}.", writer=writer)
    return model


def test_hit_sequences_hit_written_rows():
    model = _model_with_docs()
    seqs = hit_sequences(model, n=4)
    assert len(seqs) == len(set(model.memory.row_ids()))  # covering-set: проба на каждую строку
    written = set(model.memory.row_ids())
    for seq in seqs:
        rows = {TinyModel.row_hash(seq[i - 1], seq[i], model.spec.n_rows) for i in range(1, len(seq))}
        assert rows & written, "hit-проба обязана бить в строку патча"


def test_checksum_detects_in_memory_tamper():
    model = _model_with_docs()
    row_id = next(iter(model.memory.row_ids()))
    assert checksum_detects_in_memory_tamper(model, row_id)
    # значение восстановлено
    assert row_id in model.memory


def test_verify_catches_partial_clear_with_reference():
    """Эмуляция бага clear (половина строк остаётся): reference-сравнение ловит."""
    model = TinyModel()
    ref = TinyModel(seed=17)  # идентичные веса (тот же seed)
    writer = atomicng.HashWriter(model.spec)
    atomicng.write(model, "partial clear negative control document.", writer=writer)

    touched = sorted(model.memory.row_ids())
    backup = {r: (model.memory._rows[r].copy(), model.memory._counts[r], model.memory._last_touch[r])
              for r in touched}

    # самопроверка ДО бага: зелёная
    assert verify_deletion(model, n_probes=16, reference_model=ref).passed

    # эмуляция бага: clear удаляет только половину затронутых строк
    original_clear = model.memory.clear

    def buggy_clear():
        model.memory._rows.clear()
        model.memory._counts.clear()
        model.memory._last_touch.clear()
        model.memory._clock = 0
        for i, r in enumerate(touched):
            if i % 2 == 1:
                v, c, lt = backup[r]
                model.memory._rows[r] = v
                model.memory._counts[r] = c
                model.memory._last_touch[r] = lt

    model.memory.clear = buggy_clear  # type: ignore[method-assign]
    report = verify_deletion(model, n_probes=16, reference_model=ref)
    model.memory.clear = original_clear  # type: ignore[method-assign]

    assert report.passed is False, (
        "reference-сравнение ОБЯЗАНО ловить неполный clear — иначе проверка декоративна"
    )


def test_verify_full_clear_passes_with_reference():
    model = TinyModel()
    ref = TinyModel(seed=17)
    writer = atomicng.HashWriter(model.spec)
    atomicng.write(model, "full clear positive control document.", writer=writer)
    assert verify_deletion(model, n_probes=16, reference_model=ref).passed
