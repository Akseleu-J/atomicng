"""Тесты guards (P0): бюджеты атомарны, капы работают, дефолт — no-op."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomicng import SIDECAR_SMALL, BudgetExceeded, D1Rule, Event, MemoryPatch, WritePolicies


def _event(row=1, value=0.1, gate=0.5):
    return Event(row, np.full(SIDECAR_SMALL.row_width, value, np.float32), gate)


def test_row_budget_atomic():
    patch = MemoryPatch(SIDECAR_SMALL)
    rule = D1Rule()
    rule.apply(patch, [_event(row=1)], WritePolicies(row_budget=1))
    before = patch.checksum()
    with pytest.raises(BudgetExceeded):
        rule.apply(patch, [_event(row=2), _event(row=3)], WritePolicies(row_budget=1))
    assert patch.checksum() == before, "отказ по бюджету должен быть атомарным"


def test_write_budget_atomic():
    patch = MemoryPatch(SIDECAR_SMALL)
    rule = D1Rule()
    rule.apply(patch, [_event()], WritePolicies(write_budget=1))
    with pytest.raises(BudgetExceeded):
        rule.apply(patch, [_event()], WritePolicies(write_budget=1))


def test_hard_norm_cap_raises():
    patch = MemoryPatch(SIDECAR_SMALL)
    with pytest.raises(BudgetExceeded):
        D1Rule().apply(patch, [_event(value=10.0)], WritePolicies(hard_row_norm_cap=0.5))
    assert len(patch) == 0


def test_hot_cap_is_soft():
    patch = MemoryPatch(SIDECAR_SMALL)
    report = D1Rule().apply(patch, [_event(value=10.0)], WritePolicies(hot_row_norm_cap=0.5))
    assert len(patch) == 1 and report.hot_rows == [1]


def test_default_policies_noop():
    patch = MemoryPatch(SIDECAR_SMALL)
    report = D1Rule().apply(patch, [_event(value=10.0)])
    assert len(patch) == 1 and not report.warnings
