"""Conformance-тесты соответствия статье (arXiv:2610.31125) — см. FIDELITY.md.

Каждый тест фиксирует элемент статьи: формулы D1 буквально из уравнений, состав входа
гейта, признаки Eq. 7, число параметров writer'а, калибровка ρ, гиперпараметры, константы.
Реализация не имеет права тихо отклониться — падение любого теста = отклонение от статьи.

Torch-зависимые тесты архитектуры скипаются без torch; формула D1 и константы — чистое ядро.
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

# --------------------------------------------------------------- формулы D1


def test_d1_matches_paper_formulas_literally():
    """Δ_j⁺ = (1−w_j)Δ_j + A_j; A_j = (1/n)Σβ_a v_a; w_j = (1/n)Σβ_a; c_j⁺ = c_j + n — буквально."""
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
            f"строка {row}: D1 отклонился от формулы статьи"
        )


def test_counts_increment_by_block_size():
    """c_j⁺ = c_j + n_j (счётчик растёт на размер блока событий строки)."""
    from atomicng import SIDECAR_SMALL, D1Rule, Event

    patch = atomicng.MemoryPatch(SIDECAR_SMALL)
    width = SIDECAR_SMALL.row_width
    events = [Event(4, np.zeros(width, np.float32), 0.5) for _ in range(3)]
    D1Rule().apply(patch, events)
    assert patch.count(4) == 3


# ------------------------------------------------------- архитектура writer'а


def test_gate_input_composition():
    """β_a = σ(B_θ[x_a, Δ_j, v_a−Δ_j, log(1+c_j)]): вход B_θ = F + 2d + 1, гейт ∈ (0,1)."""
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
    assert (gates > 0).all() and (gates < 1).all(), "гейт обязан быть сигмоидой (β ∈ (0,1))"


def test_feature_composition_eq7():
    """x_a = [Ĥ_p; mean Ĥ_{p+1..p+4}; пул блока; эмбеддинг (order, head)] — F = 3·Hd + emb."""
    pytest.importorskip("torch")
    from atomicng.writers.mlp import WriterMLP

    assert paper.FEATURE_NEXT_WINDOW == 4
    w = WriterMLP(hidden_size=64, row_width=16, hidden=32, head_embed_dim=16)
    assert w.v_head[0].in_features == 3 * 64 + 16


def test_writer_param_count_conformance():
    """При нативной конфигурации (d=160, hidden 256, модель hidden ≈2500) параметров ≈ 4 075 681."""
    pytest.importorskip("torch")
    from atomicng.writers.mlp import WriterMLP

    w = WriterMLP(hidden_size=2500, row_width=paper.NATIVE_ROW_WIDTH, hidden=paper.WRITER_HIDDEN)
    count = sum(p.numel() for p in w.parameters())
    rel = abs(count - paper.WRITER_PARAM_COUNT) / paper.WRITER_PARAM_COUNT
    assert rel < 0.05, f"параметров {count} vs {paper.WRITER_PARAM_COUNT} ({rel:.1%})"


def test_targets_linear_by_default_and_rho_calibration():
    """Цели линейны (v = s_h·V_θ(x), без tanh) — как в статье; s_h: RMS целей ≈ ρ·RMS(T⁰)."""
    torch = pytest.importorskip("torch")
    from atomicng.writers.mlp import WriterMLP

    w = WriterMLP(hidden_size=32, row_width=16, hidden=32)
    assert w.bound_targets is False, "по умолчанию цели линейны (статья); tanh — opt-in"
    t0 = 0.02 * torch.randn(4096, 16)
    s = w.calibrate_to_table(t0)
    targets, _, _ = w(
        torch.randn(4, 40, 32),  # hidden states (B,T,Hd); признаки writer строит сам
        torch.randint(0, 4096, (4, 39)),
        torch.zeros(4, 39, 16),
        torch.zeros(4, 39),
    )
    ratio = float(targets.detach().pow(2).mean().sqrt()) / float(t0.pow(2).mean(dim=1).sqrt().mean())
    assert s == pytest.approx(w.target_scale)
    assert 0.002 < ratio < 0.05, f"RMS целей = {ratio:.4f}·RMS(T⁰), ожидалось ≈ ρ={paper.TARGET_RMS_RATIO}"


# ------------------------------------------------------------ гиперпараметры


def test_trainer_hyperparameters_conformance():
    """AdamW 3e-4, без weight decay, clip 1 — дефолты обоих тренеров = статья."""
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
    """Блоки 512/128, α=1 — дефолты соответствуют статье."""
    from atomicng.evidence import build_blocks

    sig = inspect.signature(build_blocks)
    assert sig.parameters["block_size"].default == paper.BLOCK_SIZE
    assert sig.parameters["query_prefix"].default == paper.QUERY_PREFIX
    assert atomicng.MemoryPatch(atomicng.SIDECAR_SMALL).decay == paper.DECAY_ALPHA_DEFAULT


def test_paper_constants_sane():
    assert paper.NATIVE_HASH_HEADS == paper.NATIVE_HEADS_BIGRAM + paper.NATIVE_HEADS_TRIGRAM
    assert paper.KEY_ORDERS == (2, 3)
    assert paper.INJECT_LAYER == 2
    assert paper.RETENTION_K16_DELTA > paper.RETENTION_K16_PLAIN  # delta-вариант удерживает лучше


# ------------------------------------------------------------------- подписи


def test_signing_roundtrip_and_tamper(tmp_path):
    try:
        import cryptography  # noqa: F401
    except ImportError:
        pytest.skip("cryptography не установлен — signing-тесты пропущены")
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
        pytest.skip("cryptography не установлен — signing-тесты пропущены")
    private_a, _ = atomicng.signing.generate_keypair()
    _, public_b = atomicng.signing.generate_keypair()
    f = tmp_path / "x.bin"
    f.write_bytes(b"data")
    atomicng.signing.sign_file(str(f), private_a)
    assert atomicng.signing.verify_file(str(f), public_b) is False
