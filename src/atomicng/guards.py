"""Политики записи (guards, P0): жёсткие лимиты прерывают запись ДО любого изменения,
мягкие (hot-row) попадают в WriteReport.warnings и в метрики.

Семантика соответствуют плану S2/S3: по умолчанию все лимиты выключены (совместимо
со статьёй), продакшн-конфиги включают их явно.
"""
from __future__ import annotations

from dataclasses import dataclass

from .exceptions import BudgetExceeded
from .memory import MemoryPatch


@dataclass(frozen=True)
class WritePolicies:
    """Лимиты одного оверлея.

    row_budget        — жёсткий потолок числа строк в патче (BudgetExceeded);
    write_budget      — жёсткий потолок суммарных записей (c_j) в патче;
    hot_row_norm_cap  — мягкий порог нормы строки: превышение → warning + метрика hot;
    hard_row_norm_cap — жёсткий потолок нормы строки: превышение → BudgetExceeded.
    """

    row_budget: int | None = None
    write_budget: int | None = None
    hot_row_norm_cap: float | None = None
    hard_row_norm_cap: float | None = None

    def check_budgets(self, patch: MemoryPatch, new_row_ids: set[int], n_events: int) -> None:
        violations: list[str] = []
        if self.row_budget is not None:
            free = self.row_budget - len(patch)
            missing = len(new_row_ids - set(patch.row_ids()))
            if missing > free:
                violations.append(
                    f"row_budget: требуется {missing} новых строк, свободно {free} (лимит {self.row_budget})"
                )
        if self.write_budget is not None:
            projected = patch.total_writes + n_events
            if projected > self.write_budget:
                violations.append(
                    f"write_budget: {projected} > лимита {self.write_budget} "
                    f"(добавьте бюджет или объедините оверлеи)"
                )
        if violations:
            raise BudgetExceeded("политики записи нарушены: " + "; ".join(violations), violations)

    def check_row_norm(self, norm: float) -> tuple[bool, str | None]:
        """(hot?, hard-нарушение) — мягкий порог даёт warning, жёсткий поднимает ошибку."""
        hot = self.hot_row_norm_cap is not None and norm > self.hot_row_norm_cap
        hard = self.hard_row_norm_cap is not None and norm > self.hard_row_norm_cap
        if hard:
            raise BudgetExceeded(
                f"hard_row_norm_cap: норма строки {norm:.4g} превысила жёсткий потолок {self.hard_row_norm_cap}",
                [f"norm={norm:.6g} > {self.hard_row_norm_cap}"],
            )
        return hot, None
