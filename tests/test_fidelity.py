"""Conformance tests of compliance with the article (arXiv:2610.31125) – see FIDELITY.md.

Each test captures the article element: formula D1 literally from equations, composition of input
gate, Eq. 7, number of writer parameters, calibration ρ, hyperparameters, constants.
Implementation has no right to quietly deviate - the fall of any test = deviation from the article.

Torch-dependent architecture tests boil without torch; formula D1 and constants are pure core.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atomicng
from atomicng import paper

# --------------------------------------------------------------- formula D1


def test_d1_matches_paper_formulas_literally():
    """Δ_j⁺ = (1−w_j)Δ_j + A_j; A_j = (1/n)Σβ_a v_a; w_j = (1/n)Σβ_a; c_j⁺ = c_j + n - literally."""
    from atomicng import SIDECAR_SMALL, D1Rule, Event, MemoryPatch

    rng = np.random.default_rng(11)
    width = SIDECAR_SMALL.row_width
    patch = MemoryPatch(SIDECAR_SMALL)
    patch.bump_clock()
    pre = {5: (0.05 * rng.standard_normal(width)).astype(np.float32),
           9: (0.05 * rng.standard_normal(width)).astype(np.float32)}
    for r, v in pre.items():
        patch._write_row(r, v)
        patch._add_count(r, 2)

    events = []
    expected: dict[int, np.ndarray] = {}
    for row, n_ev in ((5, 3), (9, 1), (12, 4)):
        betas = rng.uniform(0, 1, n_ev)
        targets = [(0.05 * rng.standard_normal(width)).astype(np.float32) for _ in range(n_ev)]
        events.extend(Event(row_id=row, target=t, gate=float(b)) for b, t in zip(betas, targets, strict=False))
        old = pre.get(row, np.zeros(width, np.float32))
        a = sum(b * t for b, t in zip(betas, targets, strict=False)) / n_ev
        w = float(betas.sum() / n_ev)
        expected[row] = (1.0 - w) * old + a

    D1Rule().apply(patch, events)
    for row, new in expected.items():
        assert np.allclose(patch.get(row), new.astype(np.float32), atol=1e-6), (
            f"line {row}: D1 Deviated from Article Formula"
        )


def test_counts_increment_per_block():
    """c_j⁺ = c_j + 1 per BLOCK (author review, issue #1): D1 averages a block's events
    into a single update, so the counter grows by 1 per writing block, however many
    events hit the row."""
    from atomicng import SIDECAR_SMALL, D1Rule, Event

    patch = atomicng.MemoryPatch(SIDECAR_SMALL)
    width = SIDECAR_SMALL.row_width
    block1 = [Event(4, np.zeros(width, np.float32), 0.5) for _ in range(3)]
    D1Rule().apply(patch, block1)
    assert patch.count(4) == 1, "один блок с n событиями: c_j растёт на 1, не на n"
    block2 = [Event(4, np.ones(width, np.float32), 0.5)]
    D1Rule().apply(patch, block2)
    assert patch.count(4) == 2, "второй блок: c_j == 2"


# ------------------------------------------------------- writer-architecture


def test_gate_input_composition():
    """β_a = σ(B_θ[x_a, Δ_j, v_a−Δ_j, log(1+c_j)]): entrance_θ = F + 2d + 1, gate   (0.1))."""
    torch = pytest.importorskip("torch")
    from atomicng.writers.mlp import WriterMLP

    w = WriterMLP(hidden_size=64, row_width=16, hidden=32)
    assert w.b_head[0].in_features == (3 * 64 + 16) + 2 * 16 + 1
    targets, gates, _ = w(
        torch.randn(1, 8, 64),
        torch.randint(0, 100, (1, 7)),
        torch.zeros(1, 7, 16),
        torch.zeros(1, 7),
    )
    assert targets.shape == (1, 7, 16)
    assert (gates > 0).all() and (gates < 1).all(), "The sigmoid gate must lie strictly between 0 and 1."


def test_feature_composition_eq7():
    """x_a = [Ĥ_p; mean Ĥ_{p+1..p+4}; block pool; embedding (order, head)] — F = 3·Hd + emb."""
    pytest.importorskip("torch")
    from atomicng.writers.mlp import WriterMLP

    assert paper.FEATURE_NEXT_WINDOW == 4
    w = WriterMLP(hidden_size=64, row_width=16, hidden=32, head_embed_dim=16)
    assert w.v_head[0].in_features == 3 * 64 + 16


def test_writer_param_count_conformance():
    """The native configuration has about 4,075,681 parameters (d=160, writer hidden size 256, model hidden size about 2500)."""
    pytest.importorskip("torch")
    from atomicng.writers.mlp import WriterMLP

    w = WriterMLP(hidden_size=2500, row_width=paper.NATIVE_ROW_WIDTH, hidden=paper.WRITER_HIDDEN)
    count = sum(p.numel() for p in w.parameters())
    rel = abs(count - paper.WRITER_PARAM_COUNT) / paper.WRITER_PARAM_COUNT
    assert rel < 0.05, f"parameters {count} vs {paper.WRITER_PARAM_COUNT} ({rel:.1%})"


def test_targets_linear_by_default_and_rho_calibration():
    """Goals are linear (v = s)_h·V_θ(x), without tanh, as in the article; s_h: RMS targets ρ·RMS(T)⁰)."""
    torch = pytest.importorskip("torch")
    from atomicng.writers.mlp import WriterMLP

    w = WriterMLP(hidden_size=32, row_width=16, hidden=32)
    assert w.bound_targets is False, "Default goals are linear (article); tanh is opt-in"
    t0 = 0.02 * torch.randn(4096, 16)
    s = w.calibrate_to_table(t0)
    targets, _, _ = w(
        torch.randn(4, 40, 32),  # Hidden states (B,T,Hd) – the writer builds the features
        torch.randint(0, 4096, (4, 39)),
        torch.zeros(4, 39, 16),
        torch.zeros(4, 39),
    )
    ratio = float(targets.detach().pow(2).mean().sqrt()) / float(t0.pow(2).mean(dim=1).sqrt().mean())
    assert s == pytest.approx(w.target_scale)
    assert 0.002 < ratio < 0.05, f"RMS targets = {ratio:.4f}·RMS(T⁰), expected approximately ρ={paper.TARGET_RMS_RATIO}"


# ------------------------------------------------------------ hyperparameter


def test_trainer_hyperparameters_conformance():
    """AdamW 3e-4, no weight decay, clip 1 - defaults of both coaches."""
    pytest.importorskip("torch")
    from atomicng.writers.hf_training import HFWriterTrainer
    from atomicng.writers.mlp import WriterMLP
    from atomicng.writers.training import TorchTinyModel, WriterTrainer

    sig = inspect.signature(HFWriterTrainer.__init__)
    assert sig.parameters["lr"].default == paper.LEARNING_RATE
    assert sig.parameters["clip"].default == paper.GRAD_CLIP

    model = TorchTinyModel(vocab_size=64, hidden_size=32, n_rows=256)
    writer = WriterMLP(hidden_size=32, row_width=32, hidden=32)
    spec = atomicng.TableSpec(name="t", n_rows=256, row_width=32)
    trainer = WriterTrainer(model, writer, spec)
    assert trainer.opt.param_groups[0]["lr"] == paper.LEARNING_RATE
    assert trainer.opt.param_groups[0]["weight_decay"] == paper.WEIGHT_DECAY
    assert trainer.clip == paper.GRAD_CLIP


def test_evidence_and_decay_defaults_conformance():
    """Blocks. 512/128, α = 1 – Defaults are in accordance with the article."""
    from atomicng.evidence import build_blocks

    sig = inspect.signature(build_blocks)
    assert sig.parameters["block_size"].default == paper.BLOCK_SIZE
    assert sig.parameters["query_prefix"].default == paper.QUERY_PREFIX
    assert atomicng.MemoryPatch(atomicng.SIDECAR_SMALL).decay == paper.DECAY_ALPHA_DEFAULT


def test_paper_constants_sane():
    assert paper.NATIVE_HASH_HEADS == paper.NATIVE_HEADS_BIGRAM + paper.NATIVE_HEADS_TRIGRAM
    assert paper.KEY_ORDERS == (2, 3)
    assert paper.INJECT_LAYER == 2
    assert paper.RETENTION_K16_DELTA > paper.RETENTION_K16_PLAIN  # The delta variant holds better.


# ------------------------------------------------------------------- signature


def test_signing_roundtrip_and_tamper(tmp_path):
    try:
        import cryptography  # noqa: F401
    except ImportError:
        pytest.skip("Cryptography not installed – signing tests missed")
    private_pem, public_pem = atomicng.signing.generate_keypair()
    f = tmp_path / "overlay.dng.npz"
    f.write_bytes(b"atomicng overlay bytes")
    atomicng.signing.sign_file(str(f), private_pem)
    assert atomicng.signing.verify_file(str(f), public_pem) is True

    f.write_bytes(b"tampered bytes")
    assert atomicng.signing.verify_file(str(f), public_pem) is False

    g = tmp_path / "unsigned.dng.npz"
    g.write_bytes(b"no signature file")
    assert atomicng.signing.verify_file(str(g), public_pem) is False


def test_signing_rejects_wrong_key(tmp_path):
    try:
        import cryptography  # noqa: F401
    except ImportError:
        pytest.skip("Cryptography not installed – signing tests missed")
    private_a, _ = atomicng.signing.generate_keypair()
    _, public_b = atomicng.signing.generate_keypair()
    f = tmp_path / "x.bin"
    f.write_bytes(b"data")
    atomicng.signing.sign_file(str(f), private_a)
    assert atomicng.signing.verify_file(str(f), public_b) is False
