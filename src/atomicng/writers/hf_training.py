"""Train a writer against a Hugging Face causal language model.

A differentiable sidecar injection connects the continuation cross-entropy
loss to the writer through the D1 update chain. The model and read projection
stay frozen. Each trajectory starts with empty memory, and each group uses
one backward pass. The model exposes model.model.layers and model.model.norm.
"""
from __future__ import annotations

import torch
from torch import nn

from ..backends.hf import SidecarTable, bigram_rows
from ..exceptions import BackendError
from ..specs import TableSpec


class Block:
    """Learning block: full sequence id + query-prefix."""

    __slots__ = ("ids", "query_len")

    def __init__(self, ids: list[int], query_len: int) -> None:
        if not 0 <= query_len < len(ids):
            raise ValueError(f"query_len={query_len} is out of range for {len(ids)} tokens.")
        self.ids = ids
        self.query_len = query_len

    @property
    def tokens(self) -> list[int]:
        return self.ids


class HFWriterTrainer:
    """BPTT writer coach on top of real HF model (hooks, model frozen))."""

    def __init__(
        self,
        model,
        writer: nn.Module,
        spec: TableSpec,
        table: SidecarTable,
        *,
        inject_layer: int | None = None,
        lr: float = 3e-4,
        clip: float = 1.0,
        device: str = "cpu",
    ) -> None:
        try:
            self.layers = model.model.layers
            self.norm = model.model.norm
        except AttributeError as exc:
            raise BackendError(
                "HFWriterTrainer: model.model.layers/model.norm"
            ) from exc
        self.device = device
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in table.proj.parameters():
            p.requires_grad_(False)  # proj is frozen, but the gradient flows through it in Δ
        self.writer = writer.to(device).train()
        self.spec = spec
        self.table = table.to(device)
        self.inject_layer = min(inject_layer if inject_layer is not None else 2, len(self.layers) - 1)
        self.opt = torch.optim.AdamW(self.writer.parameters(), lr=lr, weight_decay=0.0)
        self.clip = clip
        # Current trajectory (reset to each of the)
        self._delta: dict[int, torch.Tensor] = {}
        self._counts: dict[int, int] = {}
        self._ids: list[int] | None = None
        self._hidden: torch.Tensor | None = None
        self._hooks = [
            self.layers[self.inject_layer].register_forward_pre_hook(self._inject_hook, with_kwargs=True),
            self.norm.register_forward_pre_hook(self._capture_hook, with_kwargs=True),
        ]

    # ------------------------------------------------------------------ hook

    def _inject_hook(self, module, args, kwargs):
        h = args[0]
        if self._ids is None:
            return args, kwargs
        mem = self._mem_seq(self._ids).to(h.dtype)
        if h.dim() == 3:
            mem = mem.unsqueeze(0)
        return (h + mem,) + tuple(args[1:]), kwargs

    def _mem_seq(self, ids: list[int]) -> torch.Tensor:
        """(T, hidden): read bigram rows from T⁰+Δ + projection; the gradient flows through Δ."""
        rows = bigram_rows(ids, self.spec.n_rows)
        t0 = self.table.t0
        zero = torch.zeros(self.spec.row_width, device=t0.device)
        read = torch.stack([t0[r] + self._delta.get(r, zero) for r in rows])
        pad = torch.zeros(1, self.spec.row_width, device=t0.device)
        return self.table.proj(torch.cat([pad, read], dim=0))

    def _capture_hook(self, module, args, kwargs):
        self._hidden = args[0].detach()  # for signs of writer (grad-stop))

    # -------------------------------------------------------------- teaching

    def train_group(self, trajectories: list[list[Block]]) -> float:
        """One update per group of trajectories; returns the average CE continuations (nats/token)."""
        self.opt.zero_grad(set_to_none=True)
        group_terms: list[torch.Tensor] = []            # mean_ce block / len(traj) - group normalization
        report_terms: list[tuple[torch.Tensor, int]] = []  # (mean_ce block, n tokens - for the report
        n_tokens = 0
        for traj in trajectories:
            self._delta = {}
            self._counts = {}
            for block in traj:
                self._ids = block.ids
                out = self.model(input_ids=torch.tensor([block.ids], dtype=torch.long, device=self.device))
                logits = out.logits[0].float()          # (T, V); gradient through Δ-inject
                start, end = block.query_len - 1, len(block.ids) - 1
                if end > start:
                    logprobs = torch.log_softmax(logits[start:end], dim=-1)
                    targets = torch.tensor(block.ids[start + 1 : end + 1], dtype=torch.long, device=logits.device)
                    mean_ce = -logprobs[torch.arange(end - start, device=logits.device), targets].mean()
                    group_terms.append(mean_ce / len(traj))
                    report_terms.append((mean_ce, end - start))
                    n_tokens += end - start

                self._write_block(block.ids)
        self._ids = None
        if not group_terms:
            return float("nan")
        report_mean = sum(loss * n for loss, n in report_terms) / max(n_tokens, 1)
        total = sum(group_terms)
        if not total.requires_grad:
            # Group without gradient path (e.g., one block: memory is written AFTER)
            # Predictions – Update only makes sense if there is a chain of records.
            return float(report_mean.detach())
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.writer.parameters(), self.clip)
        self.opt.step()
        return float(report_mean.detach())

    def _write_block(self, ids: list[int]) -> None:
        """Differentiable block entry: D1 block averaging in δ-dictionary of trajectory."""
        rows = bigram_rows(ids, self.spec.n_rows)
        hidden = self._hidden
        if hidden is None:
            raise BackendError("Could not capture Ĥ because the model.model.norm hook did not fire.")
        # the norm hook gives (B, T, Hd) - for batch 1 it is already (1, T, Hd); the writer himself cuts the position 0
        hd = hidden if hidden.dim() == 3 else hidden.unsqueeze(0)
        dev = hidden.device
        t_rows = torch.tensor([rows], dtype=torch.long, device=dev)
        zero_d = torch.zeros(self.spec.row_width, device=dev)
        delta_rows = torch.stack([self._delta.get(r, zero_d) for r in rows]).unsqueeze(0)
        counts_t = torch.tensor(
            [[float(self._counts.get(r, 0)) for r in rows]], dtype=torch.float32, device=dev
        )  # already (1, T-1).)
        # Ĥ is already detached. Run the writer with gradients enabled so that
        # targets/gates BPTT: loss of k+1 flows through Δ in θ writer
        targets, gates, _ = self.writer(hd, t_rows, delta_rows, counts_t)
        flat_rows = rows
        flat_t = targets.reshape(-1, self.spec.row_width)
        flat_g = gates.reshape(-1)
        for r in sorted(set(rows)):
            mask = torch.tensor([x == r for x in flat_rows], device=dev)
            a = (flat_g[mask][:, None] * flat_t[mask]).mean(dim=0)
            w = flat_g[mask].mean()
            old = self._delta.get(r, torch.zeros(self.spec.row_width, device=dev))
            self._delta[r] = (1.0 - w) * old + a
            self._counts[r] = self._counts.get(r, 0) + 1  # c_j counts BLOCKS (author review)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
