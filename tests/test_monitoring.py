"""Monitoring tests: recording metrics, bench + Regress Gate, Safety-gate Publications."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atomicng
from atomicng import MetricsRegistry, PublishConfig, TinyModel


def test_metrics_registry_records_and_exports(tmp_path):
    model = TinyModel()
    reg = MetricsRegistry()
    for i in range(3):
        atomicng.write(model, f"document number {i} with facts", metrics=reg)
    rec = reg.records[-1]
    for key in ("n_events", "mean_gate", "patch_max_norm", "events_per_s", "duration_ms"):
        assert key in rec
    summary = reg.summary()
    assert summary["n_writes"] == 3 and "events_per_s_mean" in summary

    path = tmp_path / "metrics.jsonl"
    assert reg.export_jsonl(str(path)) == 3
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_jsonl_exporter_appends(tmp_path):
    exporter = atomicng.JsonlExporter(str(tmp_path / "m.jsonl"))
    exporter({"a": 1})
    exporter({"a": 2})
    content = (tmp_path / "m.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 2


def test_bench_sane_and_regression_gate():
    report = atomicng.run_bench(lambda: TinyModel(), n_docs=50, n_probes=16, repeats=3)
    for key in ("write_events_per_s", "write_docs_per_s", "read_overhead_pct", "verify_ms"):
        assert key in report
    assert report["write_events_per_s"] > 0

    failures = atomicng.check_regression(
        {**report, "write_events_per_s": report["write_events_per_s"] / 10}, report, tolerance=1.25
    )
    assert failures, "tenfold deceleration must be caught by the regress gate"
    assert atomicng.check_regression(report, report, tolerance=1.25) == []


def test_publish_gate_passes_clean_and_rejects_hot(tmp_path):
    model = TinyModel()
    report = atomicng.publish_check(model, PublishConfig(), strict=False)
    assert report.passed and report.neutral_drift == 0.0

    model2 = TinyModel()
    big = np.full(model2.spec.row_width, 5.0, np.float32)
    atomicng.D1Rule().apply(model2.memory, [atomicng.Event(11, big, 1.0)])
    with pytest.raises(atomicng.PublishRejected):
        atomicng.publish_check(model2, PublishConfig(hot_norm_cap=1.0, max_hot_rows=0), strict=True)
