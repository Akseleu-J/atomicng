"""D1 block averaging from the paper.

For the n_j events that address row j (one BLOCK = one apply call):

    A_j = (1/n_j) · Σ_a β_a·v_a
    w_j = (1/n_j) · Σ_a β_a
    Δ_j⁺ = (1 − w_j)·Δ_j + A_j
    c_j⁺ = c_j + 1          # счётчик БЛОКОВ, записавших строку:
                            # D1 усредняет события блока в одно обновление,
                            # поэтому c_j растёт на 1 за блок (подтверждено
                            # автором метода, issue #1 в reference repo)

Updates are bounded and invariant to within-block event order. A single
event reduces to Δ⁺ = (1−β)·Δ + β·v. Write policies are checked before
applying any changes to the patch.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..guards import WritePolicies
from ..memory import MemoryPatch


@dataclass(frozen=True)
class Event:
    """Event entry: (line, goal v)_a, gate β_a ∈ [0,1])."""

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
    """Recording report (S2: Nothing written quietly))."""

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
        warn = f", warning: {len(self.warnings)}" if self.warnings else ""
        return f"Wrote {self.n_events} events to {self.n_rows} rows{warn}"

    @property
    def hot_rows(self) -> list[int]:
        return [r.row_id for r in self.rows if r.hot]


class D1Rule:
    """D1 Recording Rule: Limits are transmitted through WritePolicies (see guards))."""

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
                raise ValueError(f"gate {e.gate} outside [0, 1] for row {e.row_id}")
            if np.shape(e.target) != (width,):
                raise ValueError(f"The event target must have shape ({width},).")
            groups.setdefault(int(e.row_id), []).append(e)

        policies.check_budgets(patch, set(groups), len(events))

        # Phase 1: Calculate new values and check the caps – failure before any change.
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
            hot, _ = policies.check_row_norm(after_norm)  # slug-drop picks up the error here
            warning = (
                f"hot-row: line {row_id} norm {after_norm:.4g} > cap {policies.hot_row_norm_cap}"
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

        # Phase 2: Atomic fixation. c_j растёт на 1 за блок (по статье: счётчик блоков).
        for row_id, after, row_report, warning in updates:
            patch._write_row(row_id, after)
            patch._add_count(row_id, 1)
            if warning:
                report.warnings.append(warning)
            report.rows.append(row_report)
        return report
