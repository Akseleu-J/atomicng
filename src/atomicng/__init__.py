"""atomicng — forward-only continual writing into a language model's native n-gram memory.

Метод статьи arXiv:2610.31125: T = T⁰ + Δ, запись только forward-проходами,
очистка Δ побитово возвращает исходную модель.

Публичный API v0.2a (Core — чистый NumPy; torch-слой импортируется лениво):
    model = atomicng.TinyModel()                      # бэкенд для CI; HF — через atomicng.backends.hf
    report = atomicng.write(model, evidence, policies=..., metrics=...)
    model.memory.save_pretrained("out.atomicng")
    atomicng.publish_check(model)                     # safety-gate перед публикацией
"""
import time

from . import specs
from .backends.tiny import TinyModel
from .bench import check_regression, run_bench, save_baseline
from .benchmarks import acquisition, chain, distraction, run_suite
from .evidence import Block, ByteTokenizer, build_blocks, build_blocks_from_texts
from .exceptions import (
    AtomicNGError,
    BackendError,
    BudgetExceeded,
    GuardViolation,
    OverlayError,
    PublishRejected,
    SpecError,
)
from .exporters import export_ple, load_ple
from .guards import WritePolicies
from .journal import AuditLog, UndoJournal
from .memory import MemoryPatch
from .metrics import JsonlExporter, MetricsRegistry
from .monitor_html import build_report, load_metrics_jsonl, write_report
from .publishing import PublishConfig, publish_check
from .rules.d1 import D1Rule, Event, WriteReport
from .signing import generate_keypair, sign_file, verify_file
from .specs import SIDECAR_SMALL, TableSpec
from .testing import HashWriter
from .verify import VerifyReport, verify_deletion
from .vllm_bridge import export_for_vllm, merged_table, vllm_general_plugins

__version__ = "1.0.0"

__all__ = [
    "AuditLog",
    "AtomicNGError",
    "BackendError",
    "Block",
    "BudgetExceeded",
    "ByteTokenizer",
    "D1Rule",
    "Event",
    "GuardViolation",
    "HashWriter",
    "JsonlExporter",
    "MemoryPatch",
    "MetricsRegistry",
    "OverlayError",
    "PublishConfig",
    "PublishRejected",
    "SIDECAR_SMALL",
    "SpecError",
    "TableSpec",
    "TinyModel",
    "UndoJournal",
    "VerifyReport",
    "WritePolicies",
    "WriteReport",
    "acquisition",
    "attach_memory",
    "build_blocks",
    "build_blocks_from_texts",
    "build_report",
    "export_for_vllm",
    "chain",
    "check_regression",
    "clear_memory",
    "distraction",
    "export_ple",
    "generate_keypair",
    "load_metrics_jsonl",
    "load_ple",
    "merged_table",
    "publish_check",
    "revert_last",
    "run_bench",
    "run_suite",
    "save_baseline",
    "sign_file",
    "specs",
    "verify_deletion",
    "verify_file",
    "vllm_general_plugins",
    "write",
    "write_report",
    "__version__",
]


def attach_memory(model, spec: TableSpec | None = None):
    """Подключает разреженный патч Δ к модели (у TinyModel она уже есть)."""
    if getattr(model, "memory", None) is None:
        model.memory = MemoryPatch(spec or model.spec)
    return model


def write(
    model,
    evidence,
    writer: HashWriter | None = None,
    rule: D1Rule | None = None,
    policies: WritePolicies | None = None,
    audit: AuditLog | None = None,
    undo: UndoJournal | None = None,
    metrics: MetricsRegistry | None = None,
    source: str | None = None,
) -> WriteReport:
    """Записывает свидетельство (str или список строк) в память. Только forward-вычисления.

    policies: guards (бюджеты/капы); audit: JSONL-журнал (S4);
    undo: pre-image журнал для точного отката; metrics: реестр мониторинга;
    source: идентификатор источника в provenance оверлея.
    """
    texts = [evidence] if isinstance(evidence, str) else list(evidence)
    if not texts:
        raise ValueError("пустое свидетельство")
    writer = writer or HashWriter(model.memory.spec)
    rule = rule or D1Rule()
    events = writer.events(texts)
    if undo is not None:
        undo.record(model.memory, {e.row_id for e in events})
    t0 = time.perf_counter()
    report = rule.apply(model.memory, events, policies)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if source:
        model.memory.sources.append(source)
    if metrics is not None:
        metrics.record_write(report, model.memory.stats(), elapsed_ms)
    if audit is not None:
        audit.append(
            "write",
            {
                **report.to_dict(),
                "source": source,
                "patch_checksum": model.memory.checksum(),
                "duration_ms": round(elapsed_ms, 3),
            },
        )
    return report


def revert_last(model, undo: UndoJournal) -> list[int]:
    """Точный откат последнего блока записи (pre-image journal)."""
    return undo.revert_last(model.memory)


def clear_memory(model) -> None:
    """Kill-switch: Δ=0, модель структурно равна базовой."""
    model.memory.clear()
