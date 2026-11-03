"""WriterMLP — архитектура writer'а из статьи: два 2-слойных MLP с hidden 256.

V_θ: признаки события → цель записи v_a (row_width); цель ЛИНЕЙНА (v = s_h·V_θ(x_a)),
как в статье; ограничение tanh — opt-in стабильность для sidecar-адаптера (FIDELITY.md).
B_θ: [x_a, Δ_j, v_a − Δ_j, log(1+c_j)] → скалярный гейт β_a = σ(·) ∈ [0, 1].
Масштаб s_h калибруется по статье (ρ=0.01·RMS(T⁰_h)) методом calibrate_to_table.

Вариант λ (memory_aware) различается ОРКЕСТРАЦИЕЙ (delta-вариант кодирует блоки
последовательно, перечитывая память), архитектура голов общая (как в статье:
delta форк обучается от plain).
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
        self.bound_targets = bound_targets  # False = линейные цели (статья); True = tanh-стабилизация
        # признаки: [h_p; mean h_{p+1..p+4}; пул блока; эмбеддинг (order, head)]  (Eq. 7)
        feature_dim = 3 * hidden_size + head_embed_dim
        self.head_embed = nn.Embedding(4, head_embed_dim)  # order ∈ {0: нет, 2: биграмма, 3: триграмма}
        self.v_head = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, row_width),
        )
        # вход гейта: x_a, Δ_j, v_a − Δ_j, log(1+c_j)
        self.b_head = nn.Sequential(
            nn.Linear(feature_dim + 2 * row_width + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,          # (B, T, Hd) — вход LM-head, grad-stop выполнен здесь
        row_ids: torch.Tensor,         # (B, T-1) long — адреса строк биграмм
        delta_rows: torch.Tensor,      # (B, T-1, d) — текущие Δ_j (нулевые для новых строк)
        counts: torch.Tensor,          # (B, T-1) float — c_j
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """→ (targets (B, T-1, d), gates (B, T-1), row_ids). События всех позиций p ≥ 1."""
        h = hidden.detach()  # grad-stop на Ĥ: в признаки градиенты не текут
        b, t, hd = h.shape
        hp = h[:, 1:, :]                                                  # события с 1-й позиции
        pool = h.mean(dim=1, keepdim=True).expand(b, t - 1, hd)           # пул блока
        # среднее по следующим 4 позициям (хвост зажимается на последней): h[p+1..p+4]
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
        """Масштаб s_h по статье: RMS начальных целей = ρ·RMS(T⁰_h) (ρ=0.01).

        Практическая калибровка: сырой выход V_θ прогоняется на шуме N(0,1)
        (распределение входа в статье не специфицировано), s_h = ρ·RMS(строк T⁰) / RMS(сырых целей).
        Возвращает установленный s_h.
        """
        row_rms = t0.detach().float().pow(2).mean(dim=1).sqrt().mean()
        probe = torch.randn(512, self.v_head[0].in_features, generator=torch.Generator().manual_seed(0))
        raw_rms = self.v_head(probe).float().pow(2).mean().sqrt()
        if float(raw_rms) <= 0:
            raise ValueError("сырой RMS целей нулевой — калибровка невозможна")
        self.target_scale = float(row_rms) / float(raw_rms) * rho
        return self.target_scale
