"""atomicng bench (P1): скорость записи, оверхед чтения, время verify + регресс-гейт.

Модель-зависимая часть передаётся фабрикой; для tiny-бэкенда есть готовый раннер.
Выход — JSON-отчёт; `check_regression` сравнивает с baseline с допуском и возвращает
список нарушений (пустой = зелёный) — используется в CI как блокирующий гейт.
"""
from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np


def run_bench(
    model_factory: Callable[[], object],
    *,
    n_docs: int = 200,
    n_probes: int = 64,
    repeats: int = 5,
    seed: int = 17,
) -> dict:
    """Замеряет: write throughput (документы/события в сек), read overhead %, verify время."""
    import atomicng  # локально: избегаем цикла в __init__

    model = model_factory()
    writer = atomicng.HashWriter(model.spec)
    rng = np.random.default_rng(seed)
    docs = [
        " ".join(f"token{int(x)}" for x in rng.integers(0, model.vocab_size, size=48)) for _ in range(n_docs)
    ]

    # warmup
    atomicng.write(model, docs[0], writer=writer)

    t0 = time.perf_counter()
    reports = [atomicng.write(model, doc, writer=writer) for doc in docs]
    write_s = time.perf_counter() - t0
    n_events = sum(r.n_events for r in reports)

    # read overhead: чтение нейтральных проб с полным патчем vs с очищенным
    probe_fn = model.probe
    full_times: list[float] = []
    empty_times: list[float] = []
    snap = model.memory.snapshot()
    checksum_before = model.memory.checksum()
    for _ in range(repeats):
        t = time.perf_counter()
        probe_fn(n_probes)
        full_times.append(time.perf_counter() - t)
    model.memory.clear()
    for _ in range(repeats):
        t = time.perf_counter()
        probe_fn(n_probes)
        empty_times.append(time.perf_counter() - t)
    model.memory.restore(snap)
    assert model.memory.checksum() == checksum_before, "bench исказил патч"

    full = statistics.median(full_times)
    empty = max(statistics.median(empty_times), 1e-9)

    t0 = time.perf_counter()
    atomicng.verify_deletion(model, n_probes)
    verify_s = time.perf_counter() - t0

    return {
        "n_docs": n_docs,
        "n_events": n_events,
        "write_docs_per_s": round(n_docs / write_s, 2),
        "write_events_per_s": round(n_events / write_s, 1),
        "read_time_full_ms": round(full * 1000, 3),
        "read_time_empty_ms": round(empty * 1000, 3),
        "read_overhead_pct": round((full - empty) / empty * 100.0, 2),
        "verify_ms": round(verify_s * 1000, 3),
    }


def save_baseline(report: dict, path: str) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def check_regression(current: dict, baseline: dict, tolerance: float = 1.25) -> list[str]:
    """Регресс-гейт: write медленнее baseline×tol или оверхед чтения вырос сверх baseline×tol → нарушение."""
    failures: list[str] = []
    for key in ("write_events_per_s", "write_docs_per_s"):
        limit = baseline[key] / tolerance
        if current[key] < limit:
            failures.append(f"{key}: {current[key]} < {limit:.2f} (baseline {baseline[key]}, допуск x{tolerance})")
    limit_overhead = max(baseline["read_overhead_pct"], 1.0) * tolerance
    if current["read_overhead_pct"] > limit_overhead:
        failures.append(
            f"read_overhead_pct: {current['read_overhead_pct']} > {limit_overhead:.2f} (baseline {baseline['read_overhead_pct']})"
        )
    return failures
