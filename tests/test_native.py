"""Track B: нативный read-out — конформность размерностям и свойствам статьи."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atomicng
from atomicng.native import NativeReadOut, NativeTableSpec, native_head_rows, native_rows_all_heads


def test_spec_conformance_to_paper():
    spec = NativeTableSpec()
    assert spec.row_width == 160
    assert spec.n_heads == 16
    assert spec.bigram_heads == 8 and spec.trigram_heads == 8
    assert spec.concat_dim == 2560
    assert spec.inject_layer == 2
    with pytest.raises(ValueError):
        NativeTableSpec(row_width=100)  # ширина зафиксирована статьёй


def test_head_rows_deterministic_and_separated():
    spec = NativeTableSpec(n_rows=4096)
    ids = [11, 29, 3, 47, 11, 29, 3, 47]
    big0 = native_head_rows(ids, spec, 0)
    tri0 = native_head_rows(ids, spec, spec.bigram_heads)
    assert big0 == native_head_rows(ids, spec, 0)  # детерминизм
    assert big0[0] is None and tri0[0] is None and tri0[1] is None  # прогрев ключа
    assert all(r is not None and 0 <= r < spec.n_rows for r in big0[1:])
    assert any(a != b for a, b in zip(big0[2:], tri0[2:], strict=True))  # соли голов
    allh = native_rows_all_heads(ids, spec)
    assert len(allh) == 16 and all(len(h) == len(ids) for h in allh)


def test_readout_shapes_and_paper_dims():
    spec = NativeTableSpec()
    ro = NativeReadOut(spec, out_dim=10240)
    b, t = 2, 12
    rows = torch.randn(b, t, spec.concat_dim)
    out = ro(rows)
    assert out.shape == (b, t, 10240)
    assert ro.conv.weight.numel() == 10240 * 4  # depthwise: канал × kernel 4
    assert ro.conv.dilation == (3,) and ro.conv.groups == 10240


def test_gather_uses_memory():
    spec = NativeTableSpec(n_rows=1024)
    ro = NativeReadOut(spec, out_dim=256)
    ids = [7, 13, 29, 41, 53]
    patch = atomicng.MemoryPatch(atomicng.TableSpec(name="x", n_rows=1024, row_width=spec.row_width))
    width = spec.row_width
    touched = next(iter({r for h in native_rows_all_heads(ids, spec) for r in h if r is not None}))
    patch.bump_clock()
    patch._write_row(touched, torch.full((width,), 0.5))
    ro_mem = ro.gather(ids, lambda r: patch.get(r) if r in patch else None)
    ro_base = ro.gather(ids, lambda r: None)
    assert torch.equal(ro_base[0], torch.zeros(spec.concat_dim))  # прогрев: нули
    assert not torch.equal(ro_mem[3], ro_base[3])  # Δ меняет чтение
    for i in range(len(ids)):
        for h in range(spec.n_heads):
            r = native_rows_all_heads(ids, spec)[h][i]
            if r is None or r != touched:
                continue
            diff = (ro_mem[i].reshape(spec.n_heads, width)[h] - ro_base[i].reshape(spec.n_heads, width)[h])
            assert torch.allclose(diff, torch.as_tensor(patch.get(touched)), atol=1e-6)


def test_floor_compresses_range():
    from atomicng.native import NativeReadOut

    k = torch.tensor([[1e4, -1e-4, 3.0, -2.0]])
    floored = NativeReadOut.sign_sqrt_max_floor(k)
    assert floored.abs().max() <= 1.0 + 1e-6
    assert (torch.sign(floored) == torch.sign(k)).all()
