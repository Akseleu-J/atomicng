"""Check write budgets and row norm caps before applying a patch update.

Hard limits reject the entire block. Soft norm caps produce hot-row warnings
and metrics. Limits are disabled by default and can be enabled per policy.
"""
from __future__ import annotations

from dataclasses import dataclass

from .exceptions import BudgetExceeded
from .memory import MemoryPatch


@dataclass(frozen=True)
class WritePolicies:
    """Limits of one overlay.

    row_budget        — The hard ceiling of the number of lines in the patch (BudgetExceeded));
    write_budget      — The hard ceiling of the total records (c)_j) patchy;
    hot_row_norm_cap  — The soft threshold of the norm of the line: exceeding the warning + hot-metric;
    hard_row_norm_cap — Higher Ceiling of the Standard Line: Exceeding BudgetExceeded.
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
                    f"row_budget: {missing} new rows required, {free} available (limit {self.row_budget})"
                )
        if self.write_budget is not None:
            projected = patch.total_writes + n_events
            if projected > self.write_budget:
                violations.append(
                    f"write_budget: {projected} > limit {self.write_budget} "
                    f"(Increase the budget or merge overlays)"
                )
        if violations:
            raise BudgetExceeded("Write policy violations: " + "; ".join(violations), violations)

    def check_row_norm(self, norm: float) -> tuple[bool, str | None]:
        """Report hot rows above the soft cap and reject rows above the hard cap."""
        hot = self.hot_row_norm_cap is not None and norm > self.hot_row_norm_cap
        hard = self.hard_row_norm_cap is not None and norm > self.hard_row_norm_cap
        if hard:
            raise BudgetExceeded(
                f"hard_row_norm_cap: row norm {norm:.4g} exceeds {self.hard_row_norm_cap}",
                [f"norm={norm:.6g} > {self.hard_row_norm_cap}"],
            )
        return hot, None
