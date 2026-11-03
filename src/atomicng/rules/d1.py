"""Правило записи D1 (block averaging) из статьи, адаптация DeltaNet.

На блок событий, адресующих строку j:
    A_j = (1/n_j) · Σ_a β_a·v_a          (среднее β-взвешенных целей)
    w_j = (1/n_j) · Σ_a β_a  ∈ [0, 1]    (средний гейт)
    Δ_j⁺ = (1 − w_j)·Δ_j + A_j           (выпуклая комбинация)
    c_j⁺ = c_j + n_j

Следствия (проверяются property-тестами):
- ограниченность: ‖Δ_j⁺‖ ≤ max(‖Δ_j‖, max_a ‖v_a‖)  (Prop. 1 статьи);
- порядок событий внутри блока не влияет на результат;
- одинокой записи с гейтом β соответствует Δ⁺ = (1−β)·Δ + β·v.

Бюджеты (WritePolicies) проверяются ДО любого изменения патча: нарушение — атомарный отказ.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..guards import WritePolicies
from ..memory import MemoryPatch


@dataclass(frozen=True)
class Event:
    """Событие записи: (строка, цель v_a, гейт β_a ∈ [0,1])."""

    row_id: int
    target: np.ndarray
    gate: float


@dataclass
class RowReport:
    row_id: int
    n_events: int
    w: float
    a_norm: float
    before_norm: float
    after_norm: float
    hot: bool


@dataclass
class WriteReport:
    """Отчёт о записи (S2: ничего не пишется тихо)."""

    n_events: int
    n_rows: int
    rows: list[RowReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_events": self.n_events,
            "n_rows": self.n_rows,
            "rows": [r.__dict__ for r in self.rows],
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        warn = f", предупреждений: {len(self.warnings)}" if self.warnings else ""
        return f"записано {self.n_events} событий в {self.n_rows} строк{warn}"

    @property
    def hot_rows(self) -> list[int]:
        return [r.row_id for r in self.rows if r.hot]


class D1Rule:
    """Правило записи D1. Лимиты передаются через WritePolicies (см. guards)."""

    def apply(
        self,
        patch: MemoryPatch,
        events: Sequence[Event],
        policies: WritePolicies | None = None,
    ) -> WriteReport:
        policies = policies or WritePolicies()
        if not events:
            return WriteReport(n_events=0, n_rows=0)

        width = patch.spec.row_width
        groups: dict[int, list[Event]] = {}
        for e in events:
            if not 0.0 <= e.gate <= 1.0:
                raise ValueError(f"гейт {e.gate} вне [0, 1] (строка {e.row_id})")
            if np.shape(e.target) != (width,):
                raise ValueError(f"цель события должна иметь форму ({width},)")
            groups.setdefault(int(e.row_id), []).append(e)

        policies.check_budgets(patch, set(groups), len(events))

        # Фаза 1: вычислить новые значения и проверить капы — отказ ДО любого изменения.
        patch.bump_clock()
        report = WriteReport(n_events=len(events), n_rows=len(groups))
        updates: list[tuple[int, np.ndarray, RowReport, str | None]] = []
        for row_id, evs in groups.items():
            beta = np.asarray([e.gate for e in evs], dtype=np.float32)
            targets = np.stack([np.asarray(e.target, dtype=np.float32) for e in evs])
            a = (beta[:, None] * targets).mean(axis=0)
            w = float(beta.mean())

            before = patch.get(row_id)
            after = (1.0 - w) * before + a
            after_norm = float(np.linalg.norm(after))
            hot, _ = policies.check_row_norm(after_norm)  # жёсткий кап поднимает ошибку здесь
            warning = (
                f"hot-row: строка {row_id} норма {after_norm:.4g} > капа {policies.hot_row_norm_cap}"
                if hot
                else None
            )
            updates.append(
                (
                    row_id,
                    after,
                    RowReport(
                        row_id=row_id,
                        n_events=len(evs),
                        w=w,
                        a_norm=float(np.linalg.norm(a)),
                        before_norm=float(np.linalg.norm(before)),
                        after_norm=after_norm,
                        hot=hot,
                    ),
                    warning,
                )
            )

        # Фаза 2: атомарная фиксация.
        for row_id, after, row_report, warning in updates:
            patch._write_row(row_id, after)
            patch._add_count(row_id, len(groups[row_id]))
            if warning:
                report.warnings.append(warning)
            report.rows.append(row_report)
        return report
