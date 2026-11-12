"""Verify bit-exact deletion and patch restoration.

Generate neutral probes once and compare outputs on those same inputs before
and after clearing the patch. Check the restored patch with its checksum.
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
    """Reuse neutral token sequences when available. Fall back to model.probe."""
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
        message = "Bit-exact rollback verified. Neutral probe outputs and the restored patch match."
    elif not probes_ok:
        message = "Clearing Δ changed the outputs on neutral probes."
    else:
        message = "Restoring the patch changed its checksum."
    return VerifyReport(
        passed=probes_ok and restore_ok,
        n_probes=n_probes,
        max_abs_diff=diff,
        message=message,
    )
