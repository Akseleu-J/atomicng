"""Torch layer: writer (shapes and gates), BPTT trainer (loss falls), HF sidecar-attach.

Without torch, these tests are skipped (pytest.importorskip) – the core of the torch is independent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atomicng

TINY_SPEC = atomicng.TableSpec(name="tiny-train", n_rows=256, row_width=32)


def test_writer_mlp_shapes_and_gate_range():
    from atomicng.writers.mlp import WriterMLP

    w = WriterMLP(hidden_size=32, row_width=32, hidden=64)
    hidden = torch.randn(2, 12, 32)
    rows = torch.randint(0, 256, (2, 11))
    delta = torch.randn(2, 11, 32)
    counts = torch.zeros(2, 11)
    targets, gates, out_rows = w(hidden, rows, delta, counts)
    assert targets.shape == (2, 11, 32) and gates.shape == (2, 11)
    assert torch.equal(out_rows, rows)
    assert (gates >= 0).all() and (gates <= 1).all()
    assert w.bound_targets is False  # Linear goals – the default behavior of the article


def test_tanh_bounded_targets_opt_in():
    """tanh target limitation – opt-in stabilization for sidecar adapter (FIDELITY.md))."""
    from atomicng.writers.mlp import WriterMLP

    wb = WriterMLP(hidden_size=32, row_width=32, hidden=64, target_scale=0.2, bound_targets=True)
    targets, gates, _ = wb(
        torch.randn(2, 12, 32) * 50,  # Big entrances: linear mode would fly away
        torch.randint(0, 256, (2, 11)),
        torch.zeros(2, 11, 32),
        torch.zeros(2, 11),
    )
    assert targets.abs().max() <= wb.target_scale + 1e-6  # tanh-limitation of objectives


def test_trainer_loss_decreases():
    """Synthetics: “fact” repeats in each block → memory recording should reduce CE."""
    from atomicng.writers.mlp import WriterMLP
    from atomicng.writers.training import TorchTinyModel, WriterTrainer

    model = TorchTinyModel(vocab_size=64, hidden_size=32, n_rows=256, seed=17)
    writer = WriterMLP(hidden_size=32, row_width=32, hidden=64, target_scale=0.2)
    trainer = WriterTrainer(model, writer, TINY_SPEC, lr=3e-3, clip=1.0)

    fact = [5, 9, 13, 21, 33, 41]
    noise = [2, 7, 11, 19, 23, 29, 31, 37, 43, 47]
    block_ids = fact + fact + noise  # 22 tokens: bigrams of fact occupy many positions
    trajectory = [torch.tensor(block_ids).unsqueeze(0) for _ in range(4)]

    losses = [trainer.train_group([trajectory]) for _ in range(80)]
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first, f"loss hasn’t gone down.: {first:.4f} → {last:.4f}"


def test_trainer_d1_chain_is_bptt():
    """Gradients reach the writer through the D1 chain (several blocks).).

    Blocks are deterministic and contain repetitions – guarantee of common lines → gradient path.
    """
    from atomicng.writers.mlp import WriterMLP
    from atomicng.writers.training import TorchTinyModel, WriterTrainer

    model = TorchTinyModel(vocab_size=64, hidden_size=32, n_rows=256, seed=17)
    writer = WriterMLP(hidden_size=32, row_width=32, hidden=64, target_scale=0.2)
    trainer = WriterTrainer(model, writer, TINY_SPEC, lr=1e-3)
    ids = [5, 9, 13, 21, 33, 41, 5, 9, 13, 21, 33, 41, 2, 7, 11, 19]
    trajectory = [torch.tensor(ids).unsqueeze(0) for _ in range(3)]
    loss = trainer.train_group([trajectory])
    assert loss > 0
    grads = [p.grad for p in writer.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_hf_stub_attach_write_and_bit_exact_clear():
    from atomicng.backends.hf import attach_sidecar, stub_causal_lm
    from atomicng.writers.mlp import WriterMLP

    model = stub_causal_lm(hidden_size=32, vocab_size=64, n_layers=4)
    handle = attach_sidecar(model, spec=TINY_SPEC, inject_layer=1)
    ids = list(range(3, 19))
    ids_t = torch.tensor([ids], dtype=torch.long)

    def logits_with_memory():
        with handle.write_session(ids), torch.no_grad():
            return model(input_ids=ids_t).logits

    base = logits_with_memory()  # Empty patch: read from T⁰ with Δ=0.
    writer = WriterMLP(hidden_size=32, row_width=32, hidden=64, target_scale=0.3)
    report = handle.write(ids, writer)
    assert report["n_events"] == len(ids) - 1

    written = logits_with_memory()
    assert not torch.equal(base, written), "The recording didn’t affect the exit."

    handle.memory.clear()
    assert torch.equal(base, logits_with_memory()), "Δ cleaning did not return the attached base"
    handle.detach()
