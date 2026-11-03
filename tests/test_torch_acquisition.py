"""Направление статьи на мини-стенде: память, записанная ОБУЧЕННЫМ writer'ом, снижает NLL.

Это torch-аналог Δ_empty > 0 из DEV64-протокола: HashWriter не обучен, поэтому направление
проверяем на writer'е после BPTT-обучения (WriterTrainer): траектория проигрывается с записью
памяти между блоками и без неё — с памятью продолжения должны предсказываться дешевле.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atomicng
from atomicng.writers.mlp import WriterMLP
from atomicng.writers.training import TorchTinyModel, WriterTrainer

TINY_SPEC = atomicng.TableSpec(name="tiny-train", n_rows=256, row_width=32)


def _train_writer(steps: int = 100):
    model = TorchTinyModel(vocab_size=64, hidden_size=32, n_rows=256, seed=17)
    writer = WriterMLP(hidden_size=32, row_width=32, hidden=64, target_scale=0.5)
    trainer = WriterTrainer(model, writer, TINY_SPEC, lr=4e-3, clip=1.0)
    fact = [5, 9, 13, 21, 33, 41]
    noise = [2, 7, 11, 19, 23, 29, 31, 37, 43, 47]
    block_ids = fact + fact + noise
    trajectory = [torch.tensor(block_ids).unsqueeze(0) for _ in range(4)]
    losses = [trainer.train_group([trajectory]) for _ in range(steps)]
    return model, writer, trajectory, losses


def _ce(model, block: torch.Tensor, delta_rows: torch.Tensor) -> float:
    _, logits = model(block, delta_rows)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), block[:, 1:].reshape(-1)
    ).item()


def test_trained_writer_acquisition_positive():
    model, writer, trajectory, losses = _train_writer()
    assert losses[-1] < losses[0], "обучение writer'а не прошло"

    def loss_empty_memory() -> float:
        total = 0.0
        with torch.no_grad():
            for block in trajectory:
                rows = TorchTinyModel.bigram_rows(block, model.n_rows)
                total += _ce(model, block, torch.zeros(1, rows.shape[1], TINY_SPEC.row_width))
        return total / len(trajectory)

    def loss_with_memory() -> float:
        """Тот же проход, но writer пишет в память после каждого блока (Alg. 1, only-forward)."""
        delta: dict[int, torch.Tensor] = {}
        counts: dict[int, int] = {}
        total = 0.0
        with torch.no_grad():
            for block in trajectory:
                rows = TorchTinyModel.bigram_rows(block, model.n_rows)
                flat_rows = rows.reshape(-1).tolist()
                zero = torch.zeros(TINY_SPEC.row_width)
                delta_rows = torch.stack(
                    [delta.get(r, zero) for r in flat_rows]
                ).reshape(1, -1, TINY_SPEC.row_width)
                counts_t = torch.tensor([[counts.get(r, 0) for r in flat_rows]], dtype=torch.float32)
                total += _ce(model, block, delta_rows)

                hidden = model(block, delta_rows)[0]
                targets, gates, _ = writer(hidden, rows, delta_rows, counts_t)
                flat_t = targets.reshape(-1, TINY_SPEC.row_width)
                flat_g = gates.reshape(-1)
                for r in sorted(set(flat_rows)):
                    mask = torch.tensor([x == r for x in flat_rows])
                    a = (flat_g[mask][:, None] * flat_t[mask]).mean(dim=0)
                    w = flat_g[mask].mean()
                    old = delta.get(r, torch.zeros(TINY_SPEC.row_width))
                    delta[r] = (1.0 - w) * old + a
                    counts[r] = counts.get(r, 0) + int(mask.sum())
        return total / len(trajectory)

    assert loss_with_memory() < loss_empty_memory(), (
        "память от обученного writer'а должна снижать NLL траектории (Δ_empty > 0)"
    )
