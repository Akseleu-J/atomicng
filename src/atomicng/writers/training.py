"""Обучение writer'а на raw-text цели (Алгоритм 2 статьи, мини-стенд).

Протокол статьи: блоки 512 токенов с query-префиксом 128; память стартует пустой;
блок предсказывается → затем записывается; loss нормируется на группу траекторий;
backprop идёт сквозь всю цепочку записей D1 в θ writer'а (BPTT); градиент остановлен
на Ĥ; модель заморожена; AdamW 3e-4, clip 1, без weight decay.

TorchTinyModel — учебный стенд с той же математикой чтения, что у ядра (биграммный
gather T⁰+Δ), но в torch и с per-position logits для CE. Ширина строки равна hidden_size,
чтобы вектор памяти входил в residual stream сложением (упрощение стенда).
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn

from ..specs import TableSpec


class TorchTinyModel(nn.Module):
    """Замороженный псевдо-LLM для тренировки writer'а на CPU."""

    def __init__(self, vocab_size: int = 64, hidden_size: int = 32,
                 n_rows: int = 256, seed: int = 17) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.n_rows = n_rows
        self.hidden_size = hidden_size
        self.embed = nn.Parameter(0.1 * torch.randn(vocab_size, hidden_size, generator=gen), requires_grad=False)
        self.w_out = nn.Parameter(0.05 * torch.randn(hidden_size, vocab_size, generator=gen), requires_grad=False)
        self.t0 = nn.Parameter(0.02 * torch.randn(n_rows, hidden_size, generator=gen), requires_grad=False)

    @staticmethod
    def bigram_rows(input_ids: torch.Tensor, n_rows: int) -> torch.Tensor:
        """(B, T) → (B, T-1) long: детерминированные адреса строк биграмм (blake2b — совместим с ядром)."""
        ids = input_ids.detach().cpu().tolist()
        out = torch.empty((len(ids), len(ids[0]) - 1), dtype=torch.long)
        for i, seq in enumerate(ids):
            for j in range(1, len(seq)):
                digest = hashlib.blake2b(f"{seq[j-1]}:{seq[j]}".encode(), digest_size=8).digest()
                out[i, j - 1] = int.from_bytes(digest, "little") % n_rows
        return out

    def forward(self, input_ids: torch.Tensor, delta_rows: torch.Tensor):
        """delta_rows: (B, T-1, d=hidden_size).

        → (hidden (B, T, Hd) — несёт градиент через Δ, logits (B, T-1, V)).
        """
        b, t = input_ids.shape
        rows = self.bigram_rows(input_ids, self.n_rows).to(input_ids.device)
        read_seq = self.t0[rows] + delta_rows                       # память читается из T⁰+Δ
        pad = torch.zeros(b, 1, self.hidden_size, device=input_ids.device, dtype=read_seq.dtype)
        mem_seq = torch.cat([pad, read_seq], dim=1)
        h = self.embed[input_ids] + mem_seq                         # без in-place: граф чистый
        logits = h[:, :-1, :] @ self.w_out                          # предсказание t+1 из t
        return h, logits


class WriterTrainer:
    """Тренер writer'а: BPTT сквозь цепочку записей D1 (δ-память — словарь тензоров)."""

    def __init__(self, model: TorchTinyModel, writer: nn.Module, spec: TableSpec,
                 lr: float = 3e-4, clip: float = 1.0, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.writer = writer.to(device)
        self.spec = spec
        self.clip = clip
        self.device = device
        self.opt = torch.optim.AdamW(self.writer.parameters(), lr=lr, weight_decay=0.0)

    def _delta_gather(self, delta: dict[int, torch.Tensor], rows: torch.Tensor) -> torch.Tensor:
        """Сбор Δ_j по адресам (без in-place: стек сохраняет граф BPTT)."""
        width = self.spec.row_width
        zero = torch.zeros(width, device=self.device)
        flat = [delta.get(int(r), zero) for r in rows.reshape(-1).tolist()]
        return torch.stack(flat).reshape(*rows.shape, width)

    def train_group(self, trajectories: list[list[torch.Tensor]]) -> float:
        """Один апдейт на группу траекторий (статья: 12 траекторий, loss нормируется раз).

        trajectories: список траекторий; траектория — список блоков (B=1, T) input_ids.
        Память стартует пустой на каждую траекторию; блок предсказывается → пишется.
        Loss группы — сумма по блокам; ОДИН backward сквозь всю цепочку D1 (у блока 1
        градиентного пути к θ нет — память пишется после предсказания, как в статье).
        """
        self.opt.zero_grad(set_to_none=True)
        total = 0.0
        for traj in trajectories:
            delta: dict[int, torch.Tensor] = {}
            counts: dict[int, int] = {}
            for block in traj:
                block = block.to(self.device)
                rows = TorchTinyModel.bigram_rows(block, self.model.n_rows).to(self.device)
                delta_rows = self._delta_gather(delta, rows)
                counts_t = torch.tensor(
                    [[counts.get(int(rows[i, j]), 0) for j in range(rows.shape[1])]
                     for i in range(rows.shape[0])],
                    dtype=torch.float32, device=self.device,
                )
                hidden, logits = self.model(block, delta_rows)
                targets, gates, _ = self.writer(hidden, rows, delta_rows, counts_t)

                loss_k = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), block[:, 1:].reshape(-1)
                ) / len(traj)
                total = total + loss_k

                # запись блока (дифференцируемая, Alg. 1): D1 block averaging
                flat_rows = rows.reshape(-1)
                flat_targets = targets.reshape(-1, self.spec.row_width)
                flat_gates = gates.reshape(-1)
                for r in flat_rows.unique().tolist():
                    r = int(r)
                    mask = flat_rows == r
                    a = (flat_gates[mask][:, None] * flat_targets[mask]).mean(dim=0)
                    w = flat_gates[mask].mean()
                    old = delta.get(r, torch.zeros(self.spec.row_width, device=self.device))
                    delta[r] = (1.0 - w) * old + a
                    counts[r] = counts.get(r, 0) + int(mask.sum())
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.writer.parameters(), self.clip)
        self.opt.step()
        return float(total.detach())
