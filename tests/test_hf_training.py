"""HFWriterTrainer: BPTT-обучение writer'а на настоящей HF-модели (мини-прогон).

Модель заморожена, градиенты доходят до writer'а через Δ-инжект, loss конечен,
чекпоинт writer'а сохраняется/грузится. Без сети — skip (tiny-random уже кэширован).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atomicng
from atomicng.backends.hf import SidecarTable, attach_sidecar
from atomicng.specs import TableSpec
from atomicng.writers.hf_training import Block, HFWriterTrainer
from atomicng.writers.io import load_writer
from atomicng.writers.mlp import WriterMLP

MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _load_model():
    try:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(MODEL_ID)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"HF-модель недоступна: {exc}")


def _make_trainer(model, writer_hidden=32):
    hidden_size = int(model.config.hidden_size)
    spec = TableSpec(name="hf-train", n_rows=512, row_width=16)
    table = SidecarTable(spec, hidden_size, seed=3)
    writer = WriterMLP(hidden_size=hidden_size, row_width=16, hidden=writer_hidden, target_scale=0.3)
    trainer = HFWriterTrainer(
        model, writer, spec, table, inject_layer=min(1, len(model.model.layers) - 1), lr=1e-3
    )
    return trainer, writer, spec


def test_hf_trainer_steps_gradients_and_freeze():
    model = _load_model().eval()
    trainer, writer, _spec = _make_trainer(model)
    vocab = int(model.config.vocab_size)
    ids = [i % (vocab - 1) + 1 for i in (3, 11, 29, 41, 53, 7, 13, 5, 37, 2, 19, 23)]
    traj = [Block(ids, 6), Block(ids[::-1], 6)]

    loss1 = trainer.train_group([traj])
    loss2 = trainer.train_group([traj])
    assert loss1 == loss1 and loss2 == loss2  # finite (NaN != NaN)
    for p in writer.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    grads_present = any(p.grad is not None and p.grad.abs().sum() > 0 for p in writer.parameters())
    assert grads_present, "градиенты не дошли до writer'а"
    assert all(not p.requires_grad for p in model.parameters()), "модель должна быть заморожена"
    trainer.close()


def test_hf_trainer_checkpoint_roundtrip(tmp_path):
    model = _load_model().eval()
    trainer, writer, _spec = _make_trainer(model)
    vocab = int(model.config.vocab_size)
    # повторы внутри блока → общие строки между блоками → градиентный путь через Δ
    ids = [i % (vocab - 1) + 1 for i in (5, 17, 3, 29, 11, 5, 17, 3, 29, 11)]
    trainer.train_group([[Block(ids, 4), Block(ids, 4)]])
    trainer.close()

    path = str(tmp_path / "w.safetensors")
    atomicng.writers.io.save_writer(writer, path, extra={"model_id": MODEL_ID})
    loaded, config = load_writer(path)
    assert config["model_id"] == MODEL_ID
    for k, v in writer.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_sidecar_handle_nll_changes_with_memory():
    model = _load_model().eval()
    trainer, writer, spec = _make_trainer(model)
    trainer.close()
    handle = attach_sidecar(model, spec=spec, inject_layer=0)
    vocab = int(model.config.vocab_size)
    ids = [i % (vocab - 1) + 1 for i in (7, 13, 29, 41, 53, 61, 7, 13)]
    nll_empty = handle.nll([ids])
    handle.write(ids, writer)
    nll_self = handle.nll([ids])
    assert nll_empty == nll_empty and nll_self == nll_self
    assert nll_self != nll_empty, "память не изменила NLL"
    handle.detach()
