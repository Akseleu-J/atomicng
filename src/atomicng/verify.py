"""Verify bit-exact deletion and patch restoration — with negative-control support.

Levels (per u/amargolin's review on r/LocalLLM):
1. Neutral probes before/after clear are bit-identical (Δ locality on unrelated text);
2. Reference model (same architecture and seed, empty memory): outputs on **hit probes**
   (probes that target patch rows) must match the reference bit-exactly after clear —
   this catches an INCOMPLETE clear that neutral probes are blind to;
3. Restoration is checked by the patch checksum (model-independent).

Negative control in tests/test_verify_negative.py emulates a buggy clear (keeps half
the affected rows): self-comparison passes, the reference comparison fails. The check
can fail — it is not decoration.
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


def verify_deletion(model, n_probes: int = 8, reference_model=None) -> VerifyReport:
    """Verify structural rollback. reference_model (same arch/seed, empty memory)
    enables the hit-probe comparison; without it only self-comparison runs."""
    probe = _make_probe(model, n_probes)
    checksum_before = model.memory.checksum()

    hit_seqs = None
    if reference_model is not None and len(model.memory):
        from .hit_probes import hit_sequences as _hit_sequences

        hit_seqs = _hit_sequences(model, n_probes)

    neutral_before = probe(n_probes)
    snap = model.memory.snapshot()
    model.memory.clear()
    neutral_after = probe(n_probes)
    hit_after = model.run_seqs(hit_seqs) if hit_seqs else None
    model.memory.restore(snap)
    checksum_restored = model.memory.checksum()

    hit_diff = 0.0
    hit_ok = True
    if hit_seqs:
        ref_out = _ref_run(reference_model, hit_seqs)
        hit_diff = float(np.max(np.abs(hit_after - ref_out))) if hit_after.size else 0.0
        hit_ok = bool(np.array_equal(hit_after, ref_out))

    neutral_diff = float(np.max(np.abs(neutral_before - neutral_after))) if neutral_before.size else 0.0
    neutral_ok = bool(np.array_equal(neutral_before, neutral_after))
    restore_ok = checksum_restored == checksum_before

    if neutral_ok and hit_ok and restore_ok:
        message = "Bit-exact rollback verified (neutral + hit probes, restore checksum matches)."
    elif not neutral_ok:
        message = "Clearing Δ changed the outputs on neutral probes."
    elif not hit_ok:
        message = f"Hit probes differ from the reference after clear (max diff {hit_diff:.4g})."
    else:
        message = "Restoring the patch changed its checksum."
    return VerifyReport(
        passed=neutral_ok and hit_ok and restore_ok,
        n_probes=n_probes,
        max_abs_diff=max(neutral_diff, hit_diff if hit_seqs else 0.0),
        message=message,
    )


def _ref_run(reference_model, hit_seqs):
    if hasattr(reference_model, "run_seqs"):
        return reference_model.run_seqs(hit_seqs)
    return reference_model.probe(len(hit_seqs))
