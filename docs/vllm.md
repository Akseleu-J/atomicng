# AtomicNG × vLLM — нативная интеграция

AtomicNG пишет новые знания в разреженный оверлей Δ (`T = T⁰ + Δ`), не трогая веса модели.
vLLM читает n-gram таблицу при загрузке — мост доставляет оверлей в движок тремя путями.

## Путь 1 — Merged table (рекомендован сегодня, работает везде)

Собираешь полную таблицу T⁰+Δ и подмешиваешь при загрузке модели:

```sh
pip install atomicng
atomicng write --evidence docs.txt --overlay mem.atomicng.npz
atomicng export-vllm --overlay mem.atomicng.npz --base-table t0.npz \
    --model-id Qwen3.8-Flash-Next --out ./vllm-artifacts
# → vllm-artifacts/ngram_table_merged.npz + vllm_config.json (sha256-манифест)
```

`vllm_config.json` содержит всё для раннера: путь merged-таблицы, спеку (N×d, головы),
sha256, provenance. Раннер подмешивает таблицу до старта инференса — ноль оверхеда в рантайме.

## Путь 2 — general plugin (vLLM ≥ 0.6)

Пакет регистрирует entry-point `vllm.general_plugins`. При старте сервера:

```sh
pip install atomicng
export VLLM_ATOMICNG_OVERLAY=/path/mem.atomicng.npz
vllm serve <model> ...
```

Плагин без оверлея безвреден (возвращает пустой dict), с оверлеем — отдаёт провайдер
merged-таблицы: `atomicng_ngram_overlay(base_table, spec) → (T⁰+Δ, MemoryPatch)`.

## Путь 3 — upstream Engram (Track B, v1.0)

Для моделей с заводской таблицей (Qwen3.8-Flash-Next класс) — PR в vLLM: загрузка .dng-оверлея
в существующий Engram/PLE-кодопуть (`--ngram-overlay path`). Черновик issue — ниже.

---

## Черновик issue для vllm-project/vllm (вставь при создании)

**Title:** Feature: overlay support for n-gram/Engram sidecar tables (forward-only memory writes)

**Body:**
AtomicNG (Apache-2.0, arXiv:2610.31125) implements forward-only continual writing into an
n-gram sidecar table: pretrained table T⁰ stays frozen, new knowledge accumulates in a sparse
patch Δ with block-averaged DeltaNet-style updates (convex combination ⇒ bounded norms), and
clearing Δ restores the base model bit-exactly. Writes are forward-only (no gradients).

Proposal: allow serving with an n-gram overlay — `--ngram-overlay path` — that adds Δ on top
of the factory table at load time. Artifacts are data-only safetensors/npz with sha256
manifests and provenance; a general-plugin entry point (`vllm.general_plugins` →
`atomicng_ngram_overlay`) already ships in `pip install atomicng` and can supply the merged
table (T⁰+Δ) to the engine. Calibration on SmolLM2-135M sidecar and a Tesla T4/Linux
verification (72/72 tests, bit-exact restore) are published in the repo.

Use cases: continual knowledge updates without retraining; per-tenant memory overlays
(merge/revert via pre-image journal); A/B of knowledge versions by swapping overlays.

## Совместимость

- merged-таблица: float32 LE, shape (N, d) — совпадает со спекой оверлея;
- плагин не импортирует vllm (зависимость односторонняя), тесты идут без vLLM;
- калибровка на реальном vLLM — v0.4+/Linux-стенд (см. CALIBRATION.md).
