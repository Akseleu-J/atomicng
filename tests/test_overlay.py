"""Overlay Tests: Integrity Manifesto, D1 Politics Merge, Provenance."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atomicng
from atomicng import SIDECAR_SMALL, MemoryPatch


def _write_docs(model, docs):
    writer = atomicng.HashWriter(model.spec)
    for doc in docs:
        atomicng.write(model, doc, writer=writer, source=doc[:12])


def test_integrity_manifest_detects_corruption(tmp_path):
    model = atomicng.TinyModel()
    _write_docs(model, ["integrity document one", "integrity document two"])
    path = tmp_path / "ov.npz"
    model.memory.save_pretrained(str(path))

    data = dict(np.load(path, allow_pickle=False))
    data["deltas"][0, 0] += 1.0  # spoilage
    np.savez(path, **data)

    with pytest.raises(atomicng.OverlayError, match="integrity"):
        MemoryPatch.load_pretrained(str(path))


def test_merge_d1_policy_and_provenance(tmp_path):
    m1, m2 = atomicng.TinyModel(), atomicng.TinyModel()
    _write_docs(m1, ["first overlay document about kangaroo navigation"])
    _write_docs(m2, ["second overlay document about volcanic soil chemistry"])
    p1, p2 = m1.memory, m2.memory
    rows1, rows2 = set(p1.row_ids()), set(p2.row_ids())
    assert rows1 and rows2

    p1.merge(p2, gate=0.5)
    assert rows1 <= set(p1.row_ids()) and rows2 <= set(p1.row_ids())
    # empty rows in the base: Δ = gate·v (D1 with an empty base))
    for r in rows2 - rows1:
        assert np.allclose(p1.get(r), 0.5 * p2.get(r), atol=1e-6)
    assert any(s.startswith("merge@") for s in p1.sources)

    out = tmp_path / "merged.npz"
    p1.save_pretrained(str(out))
    reloaded = MemoryPatch.load_pretrained(str(out))
    assert reloaded.checksum() == p1.checksum()
    assert reloaded.sources == p1.sources


def test_merge_spec_mismatch_rejected():
    p1 = MemoryPatch(SIDECAR_SMALL)
    p2 = MemoryPatch(atomicng.TableSpec(name="other", n_rows=64, row_width=8))
    with pytest.raises(atomicng.OverlayError):
        p1.merge(p2)


def test_created_by_roundtrip(tmp_path):
    patch = MemoryPatch(SIDECAR_SMALL, created_by="unit-test")
    path = tmp_path / "ov.npz"
    patch.save_pretrained(str(path))
    assert MemoryPatch.load_pretrained(str(path)).created_by == "unit-test"
