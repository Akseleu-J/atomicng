"""Two-layer writer MLPs for row targets and update gates.

V_θ maps event features to a linear row target scaled by s_h. Optional tanh
bounding is available for the sidecar. B_θ uses the event features, current
row, target difference, and log(1+c_j) to produce a sigmoid gate.
Plain and delta encoding share the writer architecture. Their orchestration
determines whether event features include the current patch.
"""
from __future__ import annotations

import torch
from torch import nn

from ..paper import TARGET_RMS_RATIO


class WriterMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        row_width: int,
        hidden: int = 256,
        target_scale: float = 0.05,
        head_embed_dim: int = 16,
        memory_aware: bool = False,
        bound_targets: bool = False,
    ) -> None:
        super().__init__()
        self.row_width = row_width
        self.memory_aware = memory_aware
        self.target_scale = target_scale
        self.bound_targets = bound_targets  # False = linear goals (article); True = tanh-stabilization
        # sign: [h_p; mean h_{p+1..p+4}; block pool; embedding (order, head)]  (Eq. 7)
        feature_dim = 3 * hidden_size + head_embed_dim
        self.head_embed = nn.Embedding(4, head_embed_dim)  # order ∈ {0: none, 2: bigram, 3: trigram}
        self.v_head = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, row_width),
        )
        # gate entry: x_a, Δ_j, v_a − Δ_j, log(1+c_j)
        self.b_head = nn.Sequential(
            nn.Linear(feature_dim + 2 * row_width + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,          # (B, T, Hd – LM-head entrance, grad-stop executed here
        row_ids: torch.Tensor,         # (B, T-1) long - addresses of bigram lines
        delta_rows: torch.Tensor,      # (B, T-1, d) are current Δ_j (zero-line)
        counts: torch.Tensor,          # (B, T-1) float — c_j
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """→ (targets (B, T-1, d), gates (B, T-1), row_ids). Events of all positions p ≥ 1."""
        h = hidden.detach()  # grad-stop on  : in signs gradients do not flow
        b, t, hd = h.shape
        hp = h[:, 1:, :]                                                  # Events from 1st position
        pool = h.mean(dim=1, keepdim=True).expand(b, t - 1, hd)           # block
        # average for the following 4 positions (tail clamped on the last): h[p+1..p+4]
        hpad = torch.cat([h, h[:, -1:, :].expand(b, 4, hd)], dim=1)
        nxt = (hpad[:, 2 : t + 1, :] + hpad[:, 3 : t + 2, :]
               + hpad[:, 4 : t + 3, :] + hpad[:, 5 : t + 4, :]) / 4.0
        order = torch.full((b, t - 1), 2, dtype=torch.long, device=h.device)
        feats = torch.cat([hp, nxt, pool, self.head_embed(order)], dim=-1)

        raw = self.v_head(feats)
        targets = self.target_scale * (torch.tanh(raw) if self.bound_targets else raw)
        gate_in = torch.cat(
            [feats, delta_rows, targets - delta_rows, torch.log1p(counts)[..., None]], dim=-1
        )
        gates = torch.sigmoid(self.b_head(gate_in)).squeeze(-1)
        return targets, gates, row_ids

    @torch.no_grad()
    def calibrate_to_table(self, t0: torch.Tensor, rho: float = TARGET_RMS_RATIO) -> float:
        """Scale s_h RMS of initial goals = ρ·RMS(T)⁰_h) (ρ=0.01).

        Practical calibration: raw output V_θ run at noise N(0.1))
        (The distribution of the entry in the article is not specified, s_h = ρ·RMS(T⁰ rows) / RMS(raw targets).
        Returns the established s_h.
        """
        row_rms = t0.detach().float().pow(2).mean(dim=1).sqrt().mean()
        probe = torch.randn(512, self.v_head[0].in_features, generator=torch.Generator().manual_seed(0))
        raw_rms = self.v_head(probe).float().pow(2).mean().sqrt()
        if float(raw_rms) <= 0:
            raise ValueError("Cannot calibrate because the raw target RMS is zero.")
        self.target_scale = float(row_rms) / float(raw_rms) * rho
        return self.target_scale
