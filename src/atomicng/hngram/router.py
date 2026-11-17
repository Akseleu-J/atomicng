"""TopicRouter — сверхлёгкий однослойный классификатор темы с гистерезисом.

Роутер работает по скользящему окну последних токенов: признаки — усреднённые
скрытые состояния окна. Переключает активную L2-субтаблицу только при
достаточном отрыве кандидата от текущей темы (margin) и после спада кулдауна —
это защита от ping-pong-переключений на смешанных входах (thrash).

Ошибка роутера — не отказ системы: промах темы даёт промахи строк и откат к
чистому L1 (правило отката H-Ngram).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RouteDecision:
    """Решение роутера на один вызов (диагностика thrash и отката)."""

    domain: int | None          # индекс активной темы (None — «без L2»)
    name: str | None
    switched: bool              # произошло ли переключение на этом шаге
    rejected: bool              # кандидат был отброшен гистерезисом
    margin: float               # отрыв top-1 от top-2 (или от текущей темы)
    pool_size: int              # длина скользящего окна решений


class TopicRouter(nn.Module):
    """Линейная голова поверх усреднённых признаков окна + гистерезис.

    margin     — минимальный отрыв top-1 от текущей темы для переключения;
    cooldown   — сколько шагов после переключения переключать нельзя;
    window     — размер скользящего окна решений (диагностика thrash).
    """

    def __init__(
        self,
        feature_dim: int,
        domains: list[str],
        *,
        margin: float = 0.25,
        cooldown: int = 8,
        window: int = 16,
        seed: int = 17,
    ) -> None:
        super().__init__()
        if len(domains) < 1:
            raise ValueError("domains must be non-empty.")
        if margin <= 0.0:
            raise ValueError("margin must be positive.")
        self.domains = list(domains)
        self.margin = float(margin)
        self.cooldown = int(cooldown)
        gen = torch.Generator().manual_seed(seed)
        self.head = nn.Linear(feature_dim, len(domains))
        with torch.no_grad():
            self.head.weight.normal_(0.0, 0.02, generator=gen)
            self.head.bias.zero_()
        self._current: int | None = None
        self._dwell = 0
        self._cooldown_left = 0
        self._recent: deque[int] = deque(maxlen=window)
        self.n_switches = 0
        self.n_rejects = 0
        self.n_steps = 0

    def pool(self, features: torch.Tensor, last_k: int = 8) -> torch.Tensor:
        """Скользящее окно: (B, T, d) → усреднение последних last_k позиций (B, d)."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.dim() == 2:
            return features
        k = min(last_k, features.shape[1])
        return features[:, -k:, :].mean(dim=1)

    def route(self, features: torch.Tensor, last_k: int = 8) -> RouteDecision:
        """Один шаг роутера. Признаки: (B, T, d) или (B, d); батч усредняется."""
        pooled = self.pool(features, last_k).mean(dim=0)
        scores = self.head(pooled)
        top1 = int(torch.argmax(scores.detach()))
        ordered = torch.sort(scores.detach(), descending=True).values
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float("inf")

        switched = rejected = False
        if self._current is None:
            self._current = top1
            switched = True
        elif top1 == self._current:
            self._dwell += 1
        elif self._cooldown_left == 0 and margin >= self.margin:
            self._current = top1
            self._dwell = 0
            switched = True
        else:
            rejected = True
            self.n_rejects += 1

        if switched:
            self.n_switches += 1
            self._cooldown_left = self.cooldown
        elif self._cooldown_left > 0:
            self._cooldown_left -= 1
        self.n_steps += 1
        self._recent.append(self._current)

        name = self.domains[self._current] if self._current is not None else None
        return RouteDecision(self._current, name, switched, rejected, margin, len(self._recent))

    def reset(self) -> None:
        """Новая сессия: тема не определена, счётчики на нуле."""
        self._current = None
        self._dwell = 0
        self._cooldown_left = 0
        self._recent.clear()
        self.n_switches = self.n_rejects = self.n_steps = 0

    def thrashing(self, threshold: int | None = None) -> bool:
        """Диагностика ping-pong: ≥ threshold переключений в окне решений."""
        threshold = threshold or max(2, len(self._recent) // 2)
        if len(self._recent) < 2:
            return False
        recent = list(self._recent)
        changes = sum(1 for a, b in zip(recent, recent[1:], strict=False) if a != b)
        return changes >= threshold

    def stats(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "n_switches": self.n_switches,
            "n_rejects": self.n_rejects,
            "current": self.domains[self._current] if self._current is not None else None,
            "dwell": self._dwell,
            "thrash": self.thrashing(),
        }
