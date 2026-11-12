"""Article protocols: acquisition / distraction / chain - consistency and state recovery.

Metric directions (delta)>0, retention→1 depend on the trained writer and are checked
in torch tests; here is the mechanics of protocols on the kernel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atomicng
from atomicng import TinyModel

DOCS = [
    "First document about kangaroo navigation patterns in the outback.",
    "Second document about volcanic soil chemistry and minerals.",
    "Third document about tidal energy grids near the coast.",
    "Fourth document about alpine mushroom foraging seasons.",
    "Fifth document about desert astronomy and dark skies.",
    "Sixth document about arctic ice core sampling methods.",
]


def test_nll_changes_with_memory_and_is_finite():
    model = TinyModel()
    doc = "kangaroo navigation document"
    writer = atomicng.HashWriter(model.spec)
    seq = writer.tokenize(doc)  # request with bigrams of the document itself: they address the written lines
    nll_before = model.nll([seq])
    assert np.isfinite(nll_before)
    atomicng.write(model, doc, writer=writer)
    nll_after = model.nll([seq])
    assert np.isfinite(nll_after) and nll_after != nll_before, "The memory did not change the NLL document"


def test_acquisition_consistency_and_restore():
    model = TinyModel()
    checksum = model.memory.checksum()
    report = atomicng.acquisition(model, DOCS[:3])
    assert set(report) >= {"nll_empty", "nll_self", "nll_donor", "delta_empty", "delta_donor"}
    assert report["delta_empty"] == pytest.approx(report["nll_empty"] - report["nll_self"], abs=1e-6)
    assert report["delta_donor"] == pytest.approx(report["nll_donor"] - report["nll_self"], abs=1e-6)
    assert model.memory.checksum() == checksum, "protocol must restore the state"


def test_acquisition_needs_donor():
    with pytest.raises(ValueError):
        atomicng.acquisition(TinyModel(), [DOCS[0]])


def test_distraction_structure_and_restore():
    model = TinyModel()
    checksum = model.memory.checksum()
    report = atomicng.distraction(model, DOCS, ks=(0, 1, 4))
    assert set(report["gain_nats"]) == {0, 1, 4}
    assert set(report["retention_at_k"]) == {0, 1, 4}
    assert all(v is None or -1e9 < v < 1e9 for v in report["retention_at_k"].values())
    assert model.memory.checksum() == checksum


def test_chain_curves_and_restore():
    model = TinyModel()
    checksum = model.memory.checksum()
    report = atomicng.chain(model, DOCS, n_probes=16)
    assert report["n_steps"] == len(DOCS)
    assert len(report["drift_curve"]) == len(DOCS) == len(report["recall_curve"])
    assert report["final_drift"] == report["drift_curve"][-1]
    assert all(np.isfinite(report["drift_curve"]))
    assert model.memory.checksum() == checksum


def test_run_suite_keys():
    model = TinyModel()
    report = atomicng.run_suite(model, DOCS, chain_steps=4)
    assert set(report) == {"acquisition", "distraction", "chain"}
    assert report["chain"]["n_steps"] == 4
