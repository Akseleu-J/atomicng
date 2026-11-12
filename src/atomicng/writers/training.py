"""Train a writer on the small PyTorch backend.

The default protocol uses 512-token blocks with a 128-token query prefix.
Predict each block before writing it, start each trajectory with empty
memory, and normalize loss by trajectory group. Backpropagate through D1 to
the writer while stopping gradients at Ĥ. The model stays frozen.

The backend adds bigram memory rows to the residual stream and exposes
per-position logits for continuation cross-entropy.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn

from ..specs import TableSpec


class TorchTinyModel(nn.Module):
    """Frozen pseudo-LLM for CPU writer training."""

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
        """(B, T → (B, T-1) long: deterministic addresses of bigram strings (blake2b – compatible with the kernel))."""
        ids = input_ids.detach().cpu().tolist()
        out = torch.empty((len(ids), len(ids[0]) - 1), dtype=torch.long)
        for i, seq in enumerate(ids):
            for j in range(1, len(seq)):
                digest = hashlib.blake2b(f"{seq[j-1]}:{seq[j]}".encode(), digest_size=8).digest()
                out[i, j - 1] = int.from_bytes(digest, "little") % n_rows
        return out

    def forward(self, input_ids: torch.Tensor, delta_rows: torch.Tensor):
        """delta_rows: (B, T-1, d=hidden_size).

        → (hidden (B, T, Hd) – carries a gradient through Δ, logits (B, T-1, V))).
        """
        b, t = input_ids.shape
        rows = self.bigram_rows(input_ids, self.n_rows).to(input_ids.device)
        read_seq = self.t0[rows] + delta_rows                       # Read memory from T⁰+Δ.
        pad = torch.zeros(b, 1, self.hidden_size, device=input_ids.device, dtype=read_seq.dtype)
        mem_seq = torch.cat([pad, read_seq], dim=1)
        h = self.embed[input_ids] + mem_seq                         # in-place free: graph clean
        logits = h[:, :-1, :] @ self.w_out                          # prediction of t+1 from t
        return h, logits


class WriterTrainer:
    """Writer coach: BPTT through D1 (δ-memory - tensor dictionary))."""

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
        """Collection Δ_j address (no in-place: stack saves BPTT graph))."""
        width = self.spec.row_width
        zero = torch.zeros(width, device=self.device)
        flat = [delta.get(int(r), zero) for r in rows.reshape(-1).tolist()]
        return torch.stack(flat).reshape(*rows.shape, width)

    def train_group(self, trajectories: list[list[torch.Tensor]]) -> float:
        """One update per group of trajectories (article: 12 trajectories, loss is normalized once)).

        Trajectories: list of trajectories; trajectory - list of blocks (B=1, T) input_ids.
        Memory starts empty on each trajectory; block predicted.
        Loss Group: sum by blocks; ONE is backward through the entire D1 chain (block 1)
        There is no gradient path to θ - memory is written after prediction, as in the article).
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

                # block recording (differentiable, Alg. 1): D1 block averaging
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
                    counts[r] = counts.get(r, 0) + 1  # c_j counts BLOCKS (author review)
        if not total.requires_grad:
            # Group without gradient path (no common lines between blocks)) —
            # The update is pointless, as in hf._training: skip.
            return float(total.detach())
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.writer.parameters(), self.clip)
        self.opt.step()
        return float(total.detach())
