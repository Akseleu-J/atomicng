# AtomicNG

**Atomic N-gram Memory** — atomic forward-only continual writing into a language model's native
n-gram memory. «Atomic» здесь не украшение: записи атомарны (всё-или-ничего под guards), откат
атомарен (побитовый), бюджеты атомарны.
Реализация метода [arXiv:2610.31125](https://arxiv.org/abs/2610.31125): предобученная хэш-таблица
n-грамм `T⁰` заморожена, новые знания накапливаются в разреженном патче `Δ` — запись только
forward-проходами, очистка `Δ` побитово возвращает исходную модель (`T = T⁰ + Δ`).

Apache-2.0. Ядро — чистый NumPy (без ML-фреймворков); torch — опциональный слой (writer'ы, HF).

## Установка

```sh
pip install -e .            # ядро (numpy, pyyaml)
pip install -e ".[torch]"   # + writer'ы и HF-бэкенд
pip install -e ".[dev]"     # + pytest, ruff
```

## Quickstart

```python
import atomicng

model = atomicng.TinyModel()                      # CI-бэкенд; для HF: atomicng.backends.hf.attach_sidecar
policies = atomicng.WritePolicies(row_budget=100_000, hot_row_norm_cap=1.0)
metrics, audit, undo = atomicng.MetricsRegistry(), atomicng.AuditLog("audit.jsonl"), atomicng.UndoJournal()

report = atomicng.write(model, "Документ с новым фактом...", policies=policies,
                   metrics=metrics, audit=audit, undo=undo, source="doc-1")
model.memory.save_pretrained("out.atomicng.npz")  # data-only оверлей + sha256-манифест
atomicng.publish_check(model)                     # safety-gate: verify + drift + hot-rows
```

CLI (YAML-конфиги в `configs/`):

```sh
atomicng write --evidence docs.txt --overlay mem.atomicng.npz --config configs/write_default.yaml
atomicng stats mem.atomicng.npz          # строки, нормы, reuse, provenance
atomicng verify mem.atomicng.npz         # sha256-манифест целостности
atomicng merge --base a.atomicng.npz --other b.atomicng.npz --out c.atomicng.npz
atomicng publish mem.atomicng.npz --config configs/publish_strict.yaml
atomicng bench --save baseline.json && atomicng bench --docs 200 --check baseline.json
atomicng eval --evidence examples/eval_docs.txt --suite quick --json report.json
atomicng monitor mem.atomicng.npz --metrics m.jsonl          # self-contained HTML-отчёт
atomicng export-ple --overlay mem.atomicng.npz --out table.ple  # таблица для llama.cpp
atomicng smoke
```

## Безопасность (P0)

| Механизм | Гарантия |
|---|---|
| Иммутабельная T⁰ | запись физически идёт только в разреженный Δ-стор |
| Атомарные guards | row/write-бюджеты и hard-капы проверяются ДО изменения патча |
| WriteReport | каждый write возвращает отчёт: строки, гейты, нормы, hot-флаги |
| Точный undo | pre-image журнал: `atomicng.revert_last` восстанавливает байт-в-байт |
| Kill-switch | `model.memory.clear()` — мгновенный полный откат |
| Integrity | .atomicng = data-only (npz+json, никакого кода) + sha256-манифест |
| Safety-gate | публикация только при verify ✓, drift ≤ порога, hot-rows ≤ лимита |
| Audit | append-only JSONL каждого write/merge с чексуммами |

## Мониторинг

`MetricsRegistry` → JSONL (встроено), TensorBoard / W&B (лениво): гейты, нормы патча,
hot-rows, reuse_ratio, events/s — по каждой записи и агрегированно.

## Архитектура

```
src/atomicng/
├── specs.py        # версионируемые TableSpec (валидация)
├── memory.py       # MemoryPatch: Δ-стор, ленивый decay, merge, .atomicng-оверлей, integrity
├── guards.py       # WritePolicies: бюджеты/капы (атомарные)
├── rules/d1.py     # D1 block averaging (двухфазный, property-тесты)
├── evidence.py     # документ → блоки 512/128 (протокол статьи), ByteTokenizer
├── metrics.py      # MetricsRegistry + JSONL/TensorBoard/W&B
├── bench.py        # throughput/overhead + регресс-гейт
├── publishing.py   # safety-gate: verify + drift + hot-rows
├── journal.py      # audit-лог + pre-image undo
├── verify.py       # побитовый verify_deletion на нейтральных пробах
├── backends/       # tiny (CI/CPU) · hf (torch sidecar-attach, hooks)
└── writers/        # WriterMLP (plain/delta), BPTT-тренер сквозь цепочку D1
```

Ядро не зависит от torch — CPU/любое железо; writer'ы и HF-attach — опционально (`[torch]`).
JAX-путь: ядро совместимо, полный attach — v1.x (см. `../FRAMEWORK_PLAN.md`).

## Тесты и качество

```sh
python tests/run_all.py        # или: pytest -q
ruff check src tests
```

**Conformance-тесты соответствия статье** (формулы D1 буквально, гейт, Eq. 7, параметры
writer'а, ρ-калибровка — см. `FIDELITY.md`), протоколы статьи, plain-кодирование (T⁰-only),
writer-артефакты (safetensors), подписи ed25519, HTML-монитор, PLE-экспорт, torch-слой и
**настоящая HF-модель** (tiny-random-Llama: attach → write → побитовая очистка).
Итого **72 теста**; проверка соответствия статье: `pytest tests/test_fidelity.py`.

### Обучение writer'а на своей модели (v0.3.1)

```sh
python scripts/train_writer.py --model HuggingFaceTB/SmolLM2-135M \
    --corpus corpus.txt --out runs/demo --steps 30
# → runs/demo/writer.safetensors (+ provenance) и runs/demo/eval.json (Δ_empty per doc)
```

Модель заморожена; обучается только writer (~50k–4M параметров); CPU-конфиг — блоки 96/24,
GPU-конфиг — блоки 512/128 как в статье.

### Публикация

- `python -m build` → sdist+wheel, `twine check` PASSED; wheel проверен в чистом venv
  (`atomicng smoke` зелёный) — пакет готов к PyPI; job `build` в CI держит это зелёным.
- `notebooks/QUICKSTART.ipynb` — ноутбук-визитка (код ячеек исполняется в CI-порядке без jupyter).
- `Dockerfile`: python:3.12-slim, сборка валидна только при зелёном `atomicng smoke`.

## Статус

- **v0.4.0a1 (текущая)**: plain-кодирование (T⁰-only), `atomicng monitor` (HTML),
  `atomicng export-ple`, fidelity-контроль (`paper.py` + `test_fidelity.py` + `FIDELITY.md`),
  подписи артефактов, протоколы статьи, реальное обучение writer'а. История — `CHANGELOG.md`.
- **Дальше**: full-scale обучение writer'а (GPU), vLLM-ридер, native-адаптер и upstream PR
  (v1.0) — план в `../FRAMEWORK_PLAN.md`.
