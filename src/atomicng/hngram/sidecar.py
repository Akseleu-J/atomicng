"""L2Sidecar — инъекция γ·Δ_L2 в резидуальный поток TorchTinyModel.

Конвейер одного прямого прохода (draft, синхронный; асинхронный роутер —
следующий шаг):

1. базовое чтение L1: mem_seq⁰ = t0[rows] + Δ_patch (Δ_patch — патч AtomicNG,
   в draft нулевой);
2. роутер по скользящему окну базовых скрытых состояний → активный том;
3. при смене темы менеджер подгружает том с «диска» (LRU-слоты);
4. gather адресов активной схемы → Δ_L2 = w·γ ⊙ Proj(Δ_row);
5. h = embed + mem_seq⁰ + Δ_L2 → логиты.

Proj — микро-адаптер инъекции (одна матрица row_width→hidden, без bias):
вектора субтаблицы живут в собственном пространстве тома, и замороженное ядро
не обязано уметь их потреблять — выравнивание с резидуальным потоком берёт на
себя адаптер. γ zero-init: при attach выходы бит-в-бит равны базовым
(безопасное подключение без переобучения ядра); обучаются только γ и Proj.

Два режима инъекции из концепции H-Ngram:
- режим B (этот модуль): дельта в ранний слой (скрытый поток);
- режим A (kNN-LM-style, интерполяция логитов) — не реализован здесь;
  для черновика достаточно gather+lookup из SubtableManager.
"""
from __future__ import annotations

import torch
from torch import nn

from ..writers.training import TorchTinyModel
from .manager import SubtableManager


class L2Sidecar(nn.Module):
    """Приставка L2 к замороженному TorchTinyModel. Обучаемые параметры: γ."""

    def __init__(self, model: TorchTinyModel, manager: SubtableManager) -> None:
        super().__init__()
        self.model = model
        self.manager = manager
        self.hidden_size = model.hidden_size
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.05)  # ненулевой: путь градиента для γ
        self.gamma = nn.Parameter(torch.zeros(self.hidden_size))  # zero-init gate
        self.last_read = None  # L2Read последнего прохода (диагностика)

    def _l1_rows(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Адреса L1 (blake, совместимы с ядром): (B, T-1)."""
        return TorchTinyModel.bigram_rows(input_ids, self.model.n_rows).to(input_ids.device)

    def forward(self, input_ids: torch.Tensor):
        """→ (hidden (B, T, d), logits (B, T-1, V)). Совместимо с TorchTinyModel."""
        m = self.model
        b, t = input_ids.shape
        device = input_ids.device

        # 1. Базовые скрытые состояния (L1): embed + t0[rows]; Δ_patch = 0 в draft.
        rows_l1 = self._l1_rows(input_ids)
        mem0 = m.t0[rows_l1]
        pad = torch.zeros(b, 1, self.hidden_size, device=device, dtype=mem0.dtype)
        mem_seq = torch.cat([pad, mem0], dim=1)
        h0 = m.embed[input_ids] + mem_seq

        # 2–4. Роутер по окну h0 → активный том → Δ_L2.
        delta_l2 = torch.zeros(b, t - 1, self.hidden_size, device=device, dtype=h0.dtype)
        read = self.manager.read(h0, input_ids, key_order=2)
        self.last_read = read
        if read.name is not None:
            if read.values.shape[-1] != self.hidden_size:
                raise ValueError(
                    f"Subtable {read.name!r} row_width {read.values.shape[-1]} != hidden_size {self.hidden_size}."
                )
            if read.values.shape[1] != t - 1:
                raise ValueError(f"Subtable {read.name!r} key_order must be 2 for sidecar alignment.")
            delta_l2 = read.weight * self.gamma * self.proj(read.values)

        h = h0 + torch.cat([torch.zeros(b, 1, self.hidden_size, device=device, dtype=h0.dtype), delta_l2], dim=1)
        logits = h[:, :-1, :] @ m.w_out
        return h, logits

    @staticmethod
    def attach(model: TorchTinyModel, manager: SubtableManager) -> L2Sidecar:
        """Подключение внешней памяти: ядро заморожено, γ = 0 → выходы базовые."""
        for p in model.parameters():
            p.requires_grad_(False)
        return L2Sidecar(model, manager)

    def gate_norm(self) -> float:
        """‖γ‖ — диагностика: 0.0 означает «память выключена вентильным весом»."""
        return float(self.gamma.detach().norm())
