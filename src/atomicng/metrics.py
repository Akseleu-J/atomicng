"""Мониторинг (слой observability): реестр метрик записи + экспортёры.

JSONL-экспортёр встроен (без зависимостей); TensorBoard и W&B подключаются лениво —
отсутствие пакета даёт понятную BackendError, а не падение пайплайна.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

from .exceptions import BackendError
from .rules.d1 import WriteReport


class Exporter(Protocol):
    def __call__(self, record: dict) -> None: ...


class MetricsRegistry:
    """Собирает по-записные и агрегированные метрики; every write → одна запись."""

    def __init__(self) -> None:
        self.records: list[dict] = []
        self._exporters: list[Exporter] = []

    def add_exporter(self, exporter: Exporter) -> None:
        self._exporters.append(exporter)

    # --------------------------------------------------------------- запись

    def record_write(self, report: WriteReport, patch_stats: dict, duration_ms: float) -> dict:
        gates = [r.w for r in report.rows]
        record = {
            "ts": time.time(),
            "duration_ms": round(duration_ms, 3),
            "n_events": report.n_events,
            "n_rows": report.n_rows,
            "mean_gate": sum(gates) / len(gates) if gates else 0.0,
            "max_gate": max(gates, default=0.0),
            "hot_rows": report.hot_rows,
            "n_warnings": len(report.warnings),
            "patch_rows": patch_stats["n_rows"],
            "patch_total_writes": patch_stats["total_writes"],
            "patch_max_norm": patch_stats["max_norm"],
            "reuse_ratio": patch_stats["reuse_ratio"],
            "events_per_s": report.n_events / (duration_ms / 1000.0) if duration_ms > 0 else 0.0,
        }
        self.records.append(record)
        for exporter in self._exporters:
            exporter(record)
        return record

    # ------------------------------------------------------------- экспорт

    def export_jsonl(self, path: str) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(self.records)

    def summary(self) -> dict:
        if not self.records:
            return {"n_writes": 0}
        cols = ("duration_ms", "n_events", "mean_gate", "patch_max_norm", "events_per_s")
        agg: dict = {"n_writes": len(self.records), "n_hot_rows_total": sum(len(r["hot_rows"]) for r in self.records)}
        for c in cols:
            vals = [r[c] for r in self.records]
            agg[f"{c}_mean"] = sum(vals) / len(vals)
            agg[f"{c}_max"] = max(vals)
        return agg


class JsonlExporter:
    """Append-only экспортёр в JSONL (совместим с audit-логом по формату строки)."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class TensorBoardExporter:
    """Ленивый экспортёр в TensorBoard (требует torch+tensorboard, падение — понятная ошибка)."""

    def __init__(self, logdir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415
        except ImportError as exc:
            raise BackendError(
                "TensorBoardExporter требует torch и tensorboard: pip install 'atomicng[torch]' tensorboard"
            ) from exc
        self._writer = SummaryWriter(logdir)
        self._step = 0

    def __call__(self, record: dict) -> None:
        for key in ("n_events", "mean_gate", "patch_max_norm", "events_per_s", "reuse_ratio"):
            self._writer.add_scalar(f"atomicng/{key}", record[key], self._step)
        self._writer.add_scalar("atomicng/hot_rows", len(record["hot_rows"]), self._step)
        self._step += 1
        self._writer.flush()


class WandbExporter:
    """Ленивый экспортёр в W&B (как у LLaMA-Factory: один интерфейс, много бэкендов)."""

    def __init__(self, project: str, config: dict | None = None) -> None:
        try:
            import wandb  # noqa: PLC0415
        except ImportError as exc:
            raise BackendError("WandbExporter требует wandb: pip install wandb") from exc
        self._run = wandb.init(project=project, config=config or {}, reinit=True)

    def __call__(self, record: dict) -> None:
        self._run.log({f"atomicng/{k}": v for k, v in record.items() if isinstance(v, (int, float))})
