"""Check an overlay before publication.

Require bit-exact deletion verification, neutral drift within its configured
limit, and a hot-row count at or below max_hot_rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .exceptions import PublishRejected
from .verify import verify_deletion


@dataclass(frozen=True)
class PublishConfig:
    max_neutral_drift: float = 0.0
    max_hot_rows: int = 0
    hot_norm_cap: float = 1.0
    n_probes: int = 64


@dataclass
class PublishReport:
    passed: bool
    verify_message: str
    neutral_drift: float
    hot_rows: list[int] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "verify": self.verify_message,
            "neutral_drift": self.neutral_drift,
            "hot_rows": self.hot_rows,
            "checks": self.checks,
        }


def neutral_drift(model, n_probes: int) -> float:
    """Average yield shift on neutral samples: current patch vs cleared (locality)).

    Samples are generated once (neutral)_token_seqs, If the backend can) and reused.
    """
    if hasattr(model, "neutral_token_seqs"):
        seqs = model.neutral_token_seqs(n_probes)
        run = lambda: model.run_seqs(seqs)  # noqa: E731
    else:
        run = lambda: model.probe(n_probes)  # noqa: E731
    with_patch = run()
    snap = model.memory.snapshot()
    model.memory.clear()
    base = run()
    model.memory.restore(snap)
    return float(np.mean(np.abs(with_patch - base))) if with_patch.size else 0.0


def hot_rows_above(patch, cap: float) -> list[int]:
    """Return patch row IDs whose norm exceeds the cap."""
    return sorted(r for r in patch.row_ids() if float(np.linalg.norm(patch._rows[r])) > cap)


def publish_check(model, config: PublishConfig | None = None, *, strict: bool = True) -> PublishReport:
    """Run publication checks. In strict mode, raise PublishRejected if a check fails."""
    config = config or PublishConfig()
    verify = verify_deletion(model, config.n_probes)
    drift = neutral_drift(model, config.n_probes)
    hot = hot_rows_above(model.memory, config.hot_norm_cap)

    checks = {
        "verify": verify.passed,
        "drift_ok": drift <= config.max_neutral_drift,
        "hot_rows_ok": len(hot) <= config.max_hot_rows,
        "drift": drift,
        "patch_max_norm": model.memory.stats()["max_norm"],
        "n_rows": len(model.memory),
    }
    report = PublishReport(
        passed=bool(checks["verify"] and checks["drift_ok"] and checks["hot_rows_ok"]),
        verify_message=verify.message,
        neutral_drift=drift,
        hot_rows=hot,
        checks=checks,
    )
    if strict and not report.passed:
        reasons = [k for k, v in checks.items() if isinstance(v, bool) and not v]
        raise PublishRejected(f"Overlay rejected by publication checks: {', '.join(reasons)}")
    return report
