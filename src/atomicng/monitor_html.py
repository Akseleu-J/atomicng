"""Self-contained HTML-отчёт мониторинга (`atomicng monitor`): один файл, без JS-зависимостей.

Содержание: состояние патча Δ (строки, нормы, reuse, provenance), гистограмма норм строк
(inline SVG), метрики записи (из MetricsRegistry или JSONL), вердикт safety-gate (если
дана модель). Файл можно приложить к оверлею в .dng-zoo как «сертификат здоровья».
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

from .memory import MemoryPatch  # noqa: TC001 — тип в публичной сигнатуре

_CSS = """
body{font-family:Segoe UI,system-ui,sans-serif;margin:2rem;max-width:900px;color:#1a1a2e;background:#fafafa}
h1{font-size:1.4rem} h2{font-size:1.05rem;margin-top:1.6rem;border-bottom:2px solid #e4e4ef;padding-bottom:.2rem}
table{border-collapse:collapse;width:100%;font-size:.85rem}
td,th{border:1px solid #ddd;padding:.3rem .5rem;text-align:left}
th{background:#eef} code{background:#eef;padding:.1rem .3rem;border-radius:3px}
.badge{display:inline-block;padding:.15rem .6rem;border-radius:1rem;font-size:.8rem;font-weight:600}
.ok{background:#d4f4dd;color:#0a5c2e}.warn{background:#ffe9c9;color:#7a4b00}.bad{background:#ffd6d6;color:#8b0000}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:.5rem}
.kv div{background:#fff;border:1px solid #e2e2ee;border-radius:6px;padding:.5rem .7rem}
.kv b{display:block;font-size:1.05rem} .kv span{font-size:.72rem;color:#666;text-transform:uppercase}
footer{margin-top:2rem;font-size:.75rem;color:#999}
"""


def _badge(ok: bool | None, ok_text: str, bad_text: str) -> str:
    cls = "ok" if ok else "bad"
    text = ok_text if ok else bad_text
    return f'<span class="badge {cls}">{html.escape(text)}</span>'


def _norm_histogram_svg(patch: MemoryPatch, width: int = 860, height: int = 120) -> str:
    """Гистограмма норм строк патча (log-шкала) — детектор «форматных строк» визуально."""
    import numpy as np

    norms = np.asarray([float(np.linalg.norm(v)) for v in patch._rows.values()])
    if norms.size == 0:
        return "<p>патч пуст</p>"
    log_norms = np.log10(np.maximum(norms, 1e-9))
    lo, hi = float(log_norms.min()), float(log_norms.max())
    bins = 24
    hist, edges = np.histogram(log_norms, bins=bins, range=(lo - 1e-9, hi + 1e-9))
    peak = max(int(hist.max()), 1)
    bw = width / bins
    bars = []
    for i, count in enumerate(hist):
        h = int(count / peak * (height - 24))
        if count:
            bars.append(
                f'<rect x="{i * bw:.1f}" y="{height - h:.1f}" width="{bw - 2:.1f}" height="{h}" fill="#4a6cf7"/>'
                f'<text x="{i * bw + bw / 2:.1f}" y="{height - h - 4:.1f}" font-size="9" text-anchor="middle" fill="#333">{count}</text>'
            )
    return (
        f'<svg width="{width}" height="{height + 14}" role="img">'
        + "".join(bars)
        + f'<text x="0" y="{height + 12}" font-size="10" fill="#666">log₁₀‖Δ‖: {lo:.2f} … {hi:.2f}</text></svg>'
    )


def build_report(
    patch: MemoryPatch,
    *,
    metrics_records: list[dict] | None = None,
    verify_message: str | None = None,
    verify_passed: bool | None = None,
    title: str = "AtomicNG Monitor",
) -> str:
    """Собирает HTML-отчёт. metrics_records — записи MetricsRegistry (опционально)."""
    from . import __version__  # лениво: пакет к этому моменту полностью загружен

    stats = patch.stats()
    rows_rows = ""
    norms = sorted(
        ((r, float((patch._rows[r] ** 2).sum()) ** 0.5, patch.count(r)) for r in patch.row_ids()),
        key=lambda x: -x[1],
    )[:10]
    for r, norm, count in norms:
        rows_rows += f"<tr><td><code>{r}</code></td><td>{norm:.4f}</td><td>{count}</td></tr>"

    metrics_html = "<p>метрики записи не подключены</p>"
    if metrics_records:
        n = len(metrics_records)
        gates = [r.get("mean_gate", 0.0) for r in metrics_records]
        eps = [r.get("events_per_s", 0.0) for r in metrics_records]
        hot_total = sum(len(r.get("hot_rows", [])) for r in metrics_records)
        metrics_html = (
            f'<div class="kv"><div><b>{n}</b><span>записей</span></div>'
            f"<div><b>{sum(gates) / n:.3f}</b><span>средний гейт</span></div>"
            f"<div><b>{sum(eps) / n:.0f}</b><span>events/s</span></div>"
            f"<div><b>{hot_total}</b><span>hot-строк всего</span></div></div>"
        )

    verify_html = "<p>модель не подключена — verify недоступен</p>"
    if verify_message is not None:
        verify_html = _badge(verify_passed, verify_message, verify_message)

    sources = ", ".join(html.escape(s) for s in patch.sources[-8:]) or "—"
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>
<h1>{html.escape(title)} {_badge(stats["n_rows"] > 0, "патч активен", "патч пуст")}</h1>
<p>спека <code>{html.escape(patch.spec.name)}</code> · d={patch.spec.row_width} · N={patch.spec.n_rows} · decay={patch.decay}</p>

<h2>Состояние памяти Δ</h2>
<div class="kv">
<div><b>{stats["n_rows"]}</b><span>строк</span></div>
<div><b>{stats["total_writes"]}</b><span>записей</span></div>
<div><b>{stats["reuse_ratio"]:.1%}</b><span>reuse строк</span></div>
<div><b>{stats["max_norm"]:.4f}</b><span>макс. норма</span></div>
<div><b>{stats["mean_norm"]:.4f}</b><span>средняя норма</span></div>
<div><b>{stats["clock"]}</b><span>block clock</span></div>
</div>
<h2>Гистограмма норм строк (hot-row детектор)</h2>
{_norm_histogram_svg(patch)}
<h2>Топ-10 строк по норме</h2>
<table><tr><th>row id</th><th>‖Δ_j‖</th><th>c_j</th></tr>{rows_rows or "<tr><td colspan=3>—</td></tr>"}</table>

<h2>Метрики записи</h2>
{metrics_html}

<h2>Safety-gate</h2>
{verify_html}

<h2>Provenance</h2>
<p><code>{sources}</code></p>
<footer>AtomicNG {__version__} · сгенерировано {time.strftime("%Y-%m-%d %H:%M:%S")}</footer>
</body></html>"""


def write_report(path: str, **kwargs) -> str:
    """build_report → файл; возвращает путь."""
    Path(path).write_text(build_report(**kwargs), encoding="utf-8")
    return path


def load_metrics_jsonl(path: str) -> list[dict]:
    """Читает экспорт MetricsRegistry/JsonlExporter."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
