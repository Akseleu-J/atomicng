"""CLI atomicng (P2): write / stats / verify / merge / bench / publish / smoke / version.

Конфиг — YAML (config-as-code, как у LLaMA-Factory). Все команды печатают JSON-отчёты
или человекочитаемые сводки; коды выхода — контракты для CI.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import atomicng


def _load_yaml(path: str) -> dict:
    import yaml  # локально: pyyaml нужен только CLI

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _policies_from(cfg: dict) -> atomicng.WritePolicies:
    p = cfg.get("policies") or {}
    return atomicng.WritePolicies(
        row_budget=p.get("row_budget"),
        write_budget=p.get("write_budget"),
        hot_row_norm_cap=p.get("hot_row_norm_cap"),
        hard_row_norm_cap=p.get("hard_row_norm_cap"),
    )


def _spec_from(cfg: dict) -> atomicng.TableSpec:
    s = dict(cfg.get("spec") or {})
    s.setdefault("name", "sidecar-small")
    return atomicng.TableSpec(**s)


# ------------------------------------------------------------------ команды


def cmd_write(args) -> int:
    cfg = _load_yaml(args.config) if args.config else {}
    model = atomicng.TinyModel(spec=_spec_from(cfg) if cfg.get("spec") else None)
    undo = atomicng.UndoJournal()
    audit = atomicng.AuditLog(args.audit) if args.audit else None
    metrics = atomicng.MetricsRegistry()
    if args.metrics_jsonl:
        metrics.add_exporter(atomicng.JsonlExporter(args.metrics_jsonl))
    writer = atomicng.HashWriter(model.spec, gate=float((cfg.get("writer") or {}).get("gate", 0.5)))
    policies = _policies_from(cfg)

    try:
        texts = Path(args.evidence).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise atomicng.AtomicNGError(f"файл свидетельства недоступен: {args.evidence!r} ({exc})") from exc
    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        raise atomicng.AtomicNGError(f"файл свидетельства пуст: {args.evidence!r}")
    reports = []
    for i, doc in enumerate(texts):
        reports.append(
            atomicng.write(model, doc, writer=writer, policies=policies,
                      audit=audit, undo=undo, metrics=metrics, source=f"{Path(args.evidence).name}:{i}")
        )
    model.memory.save_pretrained(args.overlay)
    if args.metrics_jsonl:
        metrics.export_jsonl(args.metrics_jsonl + ".summary.json")
    summary = {
        "overlay": args.overlay,
        "docs": len(texts),
        "last_report": reports[-1].to_dict() if reports else None,
        "memory": model.memory.stats(),
        "metrics_summary": metrics.summary(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_stats(args) -> int:
    patch = atomicng.MemoryPatch.load_pretrained(args.overlay)
    print(json.dumps({**patch.stats(), "spec": patch.spec.name, "created_by": patch.created_by},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_verify(args) -> int:
    """Статическая проверка целостности оверлея; рантайм-verify — через API/publish."""
    patch = atomicng.MemoryPatch.load_pretrained(args.overlay)  # бросит OverlayError при порче
    ok = True
    print(json.dumps({"overlay": args.overlay, "integrity": "ok", "rows": len(patch),
                      "total_writes": patch.total_writes}, ensure_ascii=False))
    return 0 if ok else 1


def cmd_merge(args) -> int:
    base = atomicng.MemoryPatch.load_pretrained(args.base)
    other = atomicng.MemoryPatch.load_pretrained(args.other)
    report = base.merge(other, gate=args.gate)
    base.save_pretrained(args.out)
    print(json.dumps({"out": args.out, "merge": report, "rows": len(base)}, ensure_ascii=False, indent=2))
    return 0


def cmd_bench(args) -> int:
    report = atomicng.run_bench(lambda: atomicng.TinyModel(), n_docs=args.docs)
    if args.save:
        atomicng.save_baseline(report, args.save)
        report["saved_to"] = args.save
    if args.check:
        baseline = json.loads(Path(args.check).read_text(encoding="utf-8"))
        failures = atomicng.check_regression(report, baseline, tolerance=args.tolerance)
        report["regression_failures"] = failures
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if failures else 0
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_publish(args) -> int:
    """Safety-gate (S3) на tiny-бэкенде: verify + drift + hot-rows."""
    model = atomicng.TinyModel()
    patch = atomicng.MemoryPatch.load_pretrained(args.overlay, model.spec)
    model.memory = patch
    cfg = _load_yaml(args.config) if args.config else {}
    pc = cfg.get("publish") or {}
    config = atomicng.PublishConfig(
        max_neutral_drift=float(pc.get("max_neutral_drift", 0.0)),
        max_hot_rows=int(pc.get("max_hot_rows", 0)),
        hot_norm_cap=float(pc.get("hot_norm_cap", 1.0)),
    )
    try:
        report = atomicng.publish_check(model, config, strict=True)
    except atomicng.PublishRejected as exc:
        print(json.dumps({"published": False, "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"published": True, **report.to_dict()}, ensure_ascii=False, indent=2))
    return 0


def cmd_smoke(args) -> int:
    model = atomicng.TinyModel()
    undo = atomicng.UndoJournal()
    audit = atomicng.AuditLog()
    metrics = atomicng.MetricsRegistry()
    writer = atomicng.HashWriter(model.spec)
    docs = [
        "First document about kangaroo navigation patterns.",
        "Second document about volcanic soil chemistry.",
        "Third document about tidal energy grids.",
    ]
    query = writer.tokenize(docs[1])

    base = model.probe()
    q_base = model.logits(query)
    for doc in docs:
        report = atomicng.write(model, doc, writer=writer, audit=audit, undo=undo, metrics=metrics)
        print(f"[write] {report.summary()}")
    neutral_unchanged = np.array_equal(base, model.probe())
    query_changed = not np.array_equal(q_base, model.logits(query))
    v1 = atomicng.verify_deletion(model)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "overlay.atomicng.npz")
        model.memory.save_pretrained(path)
        reloaded = atomicng.MemoryPatch.load_pretrained(path, model.spec)
        overlay_ok = reloaded.checksum() == model.memory.checksum()
    atomicng.revert_last(model, undo)
    v2 = atomicng.verify_deletion(model)
    atomicng.clear_memory(model)
    v3 = atomicng.verify_deletion(model)
    restored = np.array_equal(base, model.probe())
    ok = neutral_unchanged and query_changed and v1.passed and v2.passed and v3.passed and overlay_ok and restored
    print(json.dumps({
        "locality": neutral_unchanged, "query_effect": query_changed,
        "verify_after_writes": v1.message, "overlay_roundtrip": overlay_ok,
        "verify_after_revert": v2.message, "verify_after_clear": v3.message,
        "base_restored": restored, "metrics": metrics.summary(), "smoke": bool(ok),
    }, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def cmd_eval(args) -> int:
    """Протоколы статьи (acquisition/distraction/chain) на tiny-бэкенде; HF — через API."""
    try:
        docs = Path(args.evidence).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise atomicng.AtomicNGError(f"файл свидетельства недоступен: {args.evidence!r} ({exc})") from exc
    docs = [t.strip() for t in docs if t.strip()]
    if len(docs) < 2:
        raise atomicng.AtomicNGError("для протоколов нужно ≥ 2 документов (строк) в файле")
    if args.suite == "acquisition":
        report = {"acquisition": atomicng.acquisition(atomicng.TinyModel(), docs)}
    elif args.suite == "distraction":
        report = {"distraction": atomicng.distraction(atomicng.TinyModel(), docs)}
    elif args.suite == "chain":
        report = {"chain": atomicng.chain(atomicng.TinyModel(), docs)}
    else:  # quick
        model = atomicng.TinyModel()
        report = atomicng.run_suite(model, docs, chain_steps=min(len(docs), 24))
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["saved_to"] = args.json
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_monitor(args) -> int:
    """HTML-отчёт мониторинга: состояние Δ, гистограмма норм, метрики, safety-gate."""
    patch = atomicng.MemoryPatch.load_pretrained(args.overlay)
    metrics = atomicng.monitor_html.load_metrics_jsonl(args.metrics) if args.metrics else None
    verify_msg, verify_ok = None, None
    model = atomicng.TinyModel(spec=patch.spec)
    model.memory = patch
    try:
        verdict = atomicng.verify_deletion(model, n_probes=32)
        verify_msg, verify_ok = verdict.message, verdict.passed
    except RuntimeError as exc:  # патч слишком плотный для нейтральных проб
        verify_msg, verify_ok = f"verify недоступен: {exc}", None
    out = args.out or f"{args.overlay}.monitor.html"
    atomicng.monitor_html.write_report(
        out, patch=patch, metrics_records=metrics,
        verify_message=verify_msg, verify_passed=verify_ok,
    )
    print(json.dumps({"report": out, "rows": len(patch), "verify": verify_ok}, ensure_ascii=False))
    return 0


def cmd_export_ple(args) -> int:
    """Экспорт полной таблицы T⁰+Δ в PLE-бинарник для llama.cpp-движков."""
    patch = atomicng.MemoryPatch.load_pretrained(args.overlay)
    model = atomicng.TinyModel(spec=patch.spec, seed=args.seed)
    integrity = atomicng.export_ple(model.t0, patch, args.out, source=f"tiny:seed{args.seed}")
    print(json.dumps({"out": args.out, "sha256": integrity, "delta_rows": len(patch)}, ensure_ascii=False))
    return 0


# ------------------------------------------------------------------- парсер


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atomicng", description="atomicng — forward-only n-gram memory")
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command", required=False)

    p = sub.add_parser("write", help="запись свидетельства в .atomicng (tiny-бэкенд)")
    p.add_argument("--evidence", required=True, help="текстовый файл: строка = документ")
    p.add_argument("--overlay", required=True)
    p.add_argument("--config", help="YAML-конфиг (spec/policies/writer)")
    p.add_argument("--audit", help="путь audit-лога JSONL")
    p.add_argument("--metrics-jsonl", help="путь экспорта метрик")
    p.set_defaults(func=cmd_write)

    p = sub.add_parser("stats", help="статистика оверлея")
    p.add_argument("overlay")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("verify", help="целостность оверлея (sha256-манифест + спека)")
    p.add_argument("overlay")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("merge", help="слияние оверлеев (D1-политика)")
    p.add_argument("--base", required=True)
    p.add_argument("--other", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gate", type=float, default=0.5)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("bench", help="скорость записи / оверхед чтения / verify")
    p.add_argument("--docs", type=int, default=200)
    p.add_argument("--save", help="сохранить baseline JSON")
    p.add_argument("--check", help="сравнить с baseline (регресс-гейт)")
    p.add_argument("--tolerance", type=float, default=1.25)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("publish", help="safety-gate: verify + drift + hot-rows")
    p.add_argument("overlay")
    p.add_argument("--config", help="YAML с секцией publish")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("eval", help="протоколы статьи: acquisition / distraction / chain")
    p.add_argument("--evidence", required=True, help="текстовый файл: строка = документ")
    p.add_argument("--suite", choices=["quick", "acquisition", "distraction", "chain"], default="quick")
    p.add_argument("--json", help="сохранить отчёт JSON")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("monitor", help="self-contained HTML-отчёт: Δ, нормы, метрики, verify")
    p.add_argument("overlay")
    p.add_argument("--metrics", help="JSONL-файл MetricsRegistry")
    p.add_argument("--out", help="путь отчёта (по умолчанию <overlay>.monitor.html)")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("export-ple", help="PLE-таблица T⁰+Δ для llama.cpp-движков")
    p.add_argument("--overlay", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=17, help="seed sidecar-T⁰ для tiny-источника")
    p.set_defaults(func=cmd_export_ple)

    p = sub.add_parser("smoke", help="end-to-end цикл на tiny-модели")
    p.set_defaults(func=cmd_smoke)

    args = parser.parse_args(argv)
    if args.version:
        print(f"atomicng {atomicng.__version__}")
        return 0
    if args.command is None:
        parser.print_help()
        return 0
    start = time.perf_counter()
    try:
        code = args.func(args)
    except atomicng.AtomicNGError as exc:
        print(f"ОШИБКА atomicng: {exc}", file=sys.stderr)
        return 2
    if code == 0:
        print(f"[{args.command}] готово за {time.perf_counter() - start:.2f}с")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
