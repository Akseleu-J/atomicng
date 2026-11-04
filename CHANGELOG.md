# Changelog

Формат — [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/); версионирование — SemVer.

## [0.9.0rc1] — 2026-11-04

**Кандидат на v1.0.0**: публичный API заморожен (golden-тест `test_api_surface`); остались
только внешние пункты — PyPI upload, Colab TPU прогон владельца, vLLM-стенд (RELEASE_CHECKLIST.md).

### Added
- **Docs-сайт** (mkdocs Material): сборка из корневых MD единым скриптом
  (`scripts/build_docs.py`), GitHub Pages workflow (`docs.yml`), nav: Home/Calibration/Fidelity/vLLM/Changelog.
- **Golden-тест публичного API**: `__all__` заморожен; любое изменение — сознательное
  (тест печатает добавленное/удалённое и требует пометки в CHANGELOG).
- `examples/README.md` — витрина примеров (ноутбуки, протоколы, обученный writer из Release).
- `RELEASE_CHECKLIST.md` — чеклист релиза v1.0.0 для владельца.

### Fixed
- Флейк `test_trainer_d1_chain_is_bptt`: guard для групп без градиентного пути +
  детерминированные блоки (см. 0.5.0a1 — теперь и в main-тренере).

## [0.5.0a2] — 2026-11-04

### Added
- **Нативный мост к vLLM** (`vllm_bridge.py`, фишка стандартов ниши — интеграция с движком):
  - entry-point `vllm.general_plugins` → `atomicng_ngram_overlay`: при старте `vllm serve`
    плагин читает `VLLM_ATOMICNG_OVERLAY` и отдаёт провайдер merged-таблицы (T⁰+Δ);
    без оверлея плагин безвреден; vllm как зависимость НЕ нужен (вызов односторонний);
  - CLI `atomicng export-vllm`: merged таблица (.npz) + `vllm_config.json` c sha256-манифестом
    и provenance (путь 1 — работает на любом движке уже сегодня);
  - `docs/vllm.md` — три пути интеграции + готовый черновик issue для upstream vLLM;
  - 6 тестов моста (merged-математика, env-провайдер, артефакты, entry-point в метаданных).
- README: бейджи (CI/license/python/Colab-TPU/vLLM), ссылки на CALIBRATION.md и one-click Colab.

## [0.5.0a1] — 2026-11-04

### Added
- **Track B, ядро нативного адаптера** (`native.py`): `NativeTableSpec` (16 голов 8+8, d=160,
  зафиксировано статьёй), `native_head_rows`/`native_rows_all_heads` (16 адресов на позицию,
  соли голов), `NativeReadOut` — read-out статьи: concat ℝ²⁵⁶⁰ → value-путь + gated key-путь
  (RMSNorm, sign/√max floor, каузальный depthwise conv 4 тапа, dilation 3) → ℝ¹⁰²⁴⁰.
  Композиция gated-пути — документированная интерпретация (FIDELITY ≈).
- **CALIBRATION.md** — живая матрица производительности из реальных прогонов: Windows-CPU,
  Kaggle-CPU, CUDA T4 (73 мс/группа), Colab 24-ядерный CPU (54 мс/группа), TPU-чип (JAX 0.11.2
  виден; XLA-стенд отработал); обучение writer'а: 860 шагов CPU → первый acquisition +0.0422.
- Результаты Colab-TPU прогоны владельца — в evidence (`kaggle-evidence/colab-tpu/`).

### Fixed
- **Флейк test_trainer_d1_chain_is_bptt**: случайные блоки могли не иметь общих строк →
  нет градиентного пути. Теперь: guard в `WriterTrainer.train_group` (пропуск групп без пути,
  синхронно с hf_training) + детерминированные блоки с повторами в тесте.
- `test_gather_uses_memory`: torch.as_tensor для numpy-значений патча.

## [0.4.0a3] — 2026-11-02

### Fixed
- **Device-баги, найденные ночными сессиями на Kaggle** (GPU Tesla T4):
  - `SidecarHandle.forward_hidden`: input_ids теперь на устройстве модели (CUDA-модель падала
    на CPU-индексах в embedding);
  - `SidecarTable.read()`: `.numpy()` только после `.cpu()` (CUDA-тензоры);
  - `SidecarTable.inject()`: read переносится на устройство proj.
- `benchmarks.acquisition`: значения больше не округляются — дельты бит-в-бит согласованы
  с NLL на любой платформе (тест сравнения дельт падал на другом BLAS из-за double-rounding).
- `test_publish_gate` — самодостаточный YAML в tmp_path (не зависит от каталога configs/ репо;
  в кернел-окружениях его нет).

### Night sessions evidence (Kaggle, 01–02.11)
- GPU (Tesla T4): CUDA-бэкенд откалиброван — BPTT-стенд loss 5.55→4.05, 73мс/группа;
  pytest 70/72 (2 падения — исправлены в этой версии).
- CPU: полный бюджет 7.8ч — **860 шагов обучения writer'а на SmolLM2-135M, loss 3.36→3.12
  (min 1.606), первый настоящий acquisition Δ_empty=+0.0422**; pytest 71/72.
- TPU: XLA-цепочка BPTT на чипе работает; сессия отменена на ~2ч (S0–S4 собраны).

## [0.4.0a2] — 2026-11-01

### Added
- **Device-поддержка ночного TPU/GPU/CPU прогона**: `HFWriterTrainer(..., device=...)` — модель,
  writer и sidecar-таблица переносятся на устройство; все тензоры тренировки (блоки, цели,
  arange, δ-словарь, маски) создаются на устройстве; `SidecarTable.to(device)`;
  `SidecarHandle.nll` корректен для моделей на любом устройстве.

### Fixed
- `SidecarHandle.nll`: индексы/arange теперь на устройстве логитов (CUDA-модели не падали раньше
  только потому, что путь не проверялся).

## [0.4.0a1] — 2026-10-31

### Added
- **Release engineering**: `python -m build` собирает sdist+wheel; `twine check` PASSED;
  wheel проверен установкой в чистый venv с запуском `atomicng smoke` (WHEEL_SMOKE_OK);
  job `build` добавлен в CI.
- **`notebooks/QUICKSTART.ipynb`**: ноутбук-визитка (базовый цикл, guards, протоколы,
  safety-gate, HTML-монитор) — готов к Colab.
- **`Dockerfile`**: python:3.12-slim, ядро без torch, `atomicng smoke` как часть сборки
  (образ считается валидным только при зелёном smoke); `.dockerignore`.
- **Plain-режим кодирования** (закрыт gap FIDELITY): `handle.write(..., memory_aware=False)` —
  plain-вариант статьи: writer кодирует свидетельство на T⁰-only модели (нулевые Δ_j/c_j во
  входе гейта), запись в память при этом обычная; delta-вариант — по умолчанию. Тест ловит
  фактические входы writer'а в обоих режимах.
- **`atomicng monitor`**: self-contained HTML-отчёт (`monitor_html.py`) — состояние Δ,
  SVG-гистограмма норм строк (визуальный hot-row детектор), метрики записи (JSONL),
  вердикт verify; без JS-зависимостей; годится как «сертификат здоровья» оверлея в .dng-zoo.
- **`atomicng export-ple`** (`exporters.py`): полная таблица T⁰+Δ в бинарный формат
  DNGPLE01 (header-json + float32 rows, sha256-integrity) для llama.cpp-движков;
  строгая загрузка (magic/integrity/спека).
- Тестов: **72** (было 66).

### Fixed
- Циклический импорт monitor_html → atomicng (версия теперь берётся лениво).

## [0.3.2a1] — 2026-10-31

### Added
- **Контроль соответствия статье (fidelity control)**:
  - `paper.py` — канонические константы статьи (512/128, группа 12, AdamW 3e-4/clip 1/wd 0,
    ρ=0.01, d=160, H=16=8+8, ≈4 075 681 параметров writer'а, эталонные Δ_empty/retention) —
    единственный источник истины;
  - `tests/test_fidelity.py` (12 тестов): формулы D1 сверяются с уравнениями статьи буквально,
    состав входа гейта, признаки Eq. 7, число параметров writer'а (±5% на нативной конфигурации),
    калибровка ρ, дефолты тренеров и evidence, дефолт α=1;
  - **FIDELITY.md** — по-элементная таблица «статья ↔ AtomicNG» (✓/≈/✗/+), реестр осознанных
    отклонений с мотивацией, clean-room политика по официальному репозиторию (нет лицензии —
    код не читался, контроль по статье + публичному README).
- **WriterMLP: соответствие статье усилено** — цели линейны по умолчанию (v = s_h·V_θ(x),
  как в статье; tanh теперь opt-in `bound_targets=True`); `calibrate_to_table(t0)` реализует
  калибровку s_h по ρ=0.01·RMS(T⁰_h).
- **Подпись артефактов (S1)**: `signing.py` — detached ed25519-подписи оверлеев и writer-чекпоинтов
  (`generate_keypair` / `sign_file` / `verify_file`); порча ловится манифестом, подмена — подписью.
- Тестов: **66** (было 54).

### Changed
- WriterMLP `target_scale` по умолчанию не ограничивает цели (tanh opt-in); конфиг writer-артефакта
  несёт `bound_targets` для точного round-trip.

## [0.3.1a1] — 2026-10-31

### Added
- **HFWriterTrainer** (`writers/hf_training.py`): обучение writer'а на настоящей HF-модели —
  Δ входит в forward дифференцируемым инжектом перед слоем L, градиент течёт СКВОЗЬ
  замороженные модель и proj в θ writer'а (BPTT сквозь цепочку D1); loss — CE на продолжении
  блока (query-префикс по протоколу статьи); группы без градиентного пути (например,
  одиночный блок) корректно пропускаются.
- **`SidecarHandle.nll`**: средний NLL (nats/token) при текущей памяти — eval-примитив
  для acquisition на настоящих моделях.
- **`scripts/train_writer.py`**: полный пайплайн v0.3.1 — корпус (абзацы) → блоки →
  BPTT-обучение → чекпоинт safetensors (+ provenance: model_id, corpus_sha256) →
  eval acquisition на удержанных документах.
- **Тесты: 54**: шаги HF-тренера на tiny-random Llama, градиенты конечны и доходят до
  writer'а, модель остаётся замороженной, чекпоинт round-trip, NLL меняется от памяти.
- Первый **обученный writer-чекпоинт AtomicNG**: SmolLM2-135M (Llama-архитектура) +
  реальный корпус (Gutenberg, public domain), CPU-mini конфиг.

### Fixed
- `no_grad` вокруг writer'а в записи блока рвал BPTT-граф (градиент не доходил до θ);
  `counts_t` лишний unsqueeze (4-D вместо 3-D); Ĥ-хук уже даёт (B,T,Hd) — двойной unsqueeze.

## [0.3.0a1] — 2026-10-31

### Added
- **Реестр протоколов статьи** (`benchmarks.py`, T5): `acquisition` (Δ_empty/Δ_donor),
  `distraction` (retention@k), `chain` (drift на нейтральных пробах + recall + stability),
  `run_suite`. Все протоколы снапшотят и восстанавливают патч, бэкенд-агностичны
  (модели нужен `nll(seqs)`).
- **NLL-интерфейс TinyModel**: `token_logits` (словарная голова `w_vocab`), `nll(seqs)`.
- **Артефакты writer'а** (`writers/io.py`): safetensors (data-only, без pickle — S1),
  конфиг архитектуры в метаданных, строгая валидация версии, `writer_fingerprint`
  (sha256 state_dict для provenance).
- **CLI**: `atomicng eval --suite quick|acquisition|distraction|chain --json report.json`;
  пример в `examples/`.
- **Тесты: 51** (было 40), в том числе:
  - протоколы: консистентность дельт, восстановление состояния, кривые цепочки;
  - writer-артефакты: round-trip, отказ на чужой версии, различимость fingerprint'ов;
  - **настоящая HF-модель** (`hf-internal-testing/tiny-random-LlamaForCausalLM`): attach →
    write → выход изменился → clear → **побитово** вернулся → overlay round-trip. Первый
    прогон backends.hf за пределами стаба (при отсутствии сети — skip);
  - **направление Δ_empty > 0**: память от writer'а после BPTT-обучения снижает NLL
    траектории против пустой памяти (на мини-стенде TorchTinyModel).

### Changed
- torch-extra теперь включает `safetensors>=0.4`.

## [0.2.0a2] — 2026-10-31

### Changed
- **Бренд: проект переименован `dng` → `atomicng` (AtomicNG — Atomic N-gram Memory)** по
  решению владельца: «atomic» — личный бренд и обязателен в названии. Переименованы пакет,
  импорт, CLI (`atomicng`), формат оверлея (`atomicng-overlay`), спека (`atomicng-spec`);
  README/CHANGELOG/CI/планы обновлены. Название метода статьи «Delta N-gram» при цитировании
  сохранено. Публичного API-обещания на старом имени не существовало (pre-release).

## [0.2.0a1] — 2026-10-31

### Added
- **Guards (P0)**: `WritePolicies` — row_budget / write_budget (жёсткие, атомарный отказ),
  hot_row_norm_cap (мягкий) и hard_row_norm_cap; двухфазное применение D1 (вычисление+валидация
  до фиксации).
- **Evidence-пайплайн**: `build_blocks` (512 токенов / query-префикс 128, протокол статьи),
  `ByteTokenizer` — закрывает пробел «документ → события записи», отсутствующий у конкурентов.
- **Мониторинг**: `MetricsRegistry` + экспортёры JSONL / TensorBoard / W&B (ленивые);
  по-записные метрики (гейты, нормы, hot-rows, reuse_ratio, events/s) и агрегаты.
- **Bench + регресс-гейт**: `atomicng bench` — write throughput, read overhead, verify-время;
  `check_regression` для CI.
- **Safety-gate публикации** (`atomicng publish`): verify + drift на нейтральных пробах + hot-rows;
  нейтральные пробы генерируются один раз и не адресуют строки патча (`neutral_token_seqs`).
- **Torch-слой**: `WriterMLP` (V_θ/B_θ, 2×256, гейт σ([x, Δ, v−Δ, log(1+c)]), tanh-масштаб целей),
  `TorchTinyModel` + `WriterTrainer` (BPTT сквозь цепочку D1, один backward на группу,
  AdamW 3e-4/clip 1, grad-stop на Ĥ), `backends.hf.attach_sidecar` (hooks-инжект перед слоем L,
  захват Ĥ, stub causal-LM для тестов без весов).
- **Integrity оверлеев**: sha256-манифест внутри .atomicng (порча → отказ загрузки), merge по
  D1-политике, provenance (sources), created_by.
- **CLI**: write / stats / verify / merge / bench / publish / smoke; YAML-конфиги
  (`configs/write_default.yaml`, `configs/publish_strict.yaml`).
- Исключения с подсказками (`atomicng.exceptions`), py.typed, CI (core + torch jobs), ruff.

### Changed
- verify/publish используют нейтральные пробы (на плотной sidecar-таблице случайные пробы
  легитимно задевают записанные строки — это эффект памяти, а не нарушение).
- D1Rule: лимиты перенесены в WritePolicies (конструктор чистый).

## [0.1.0] — 2026-10-31

### Added
- Core на чистом NumPy: TableSpec, MemoryPatch (ленивый decay, snapshot/restore,
  overlay round-trip), D1 (block averaging) + property-тесты (выпуклость, ограниченность,
  порядок-инвариантность), undo через pre-image журнал, audit-лог, verify_deletion,
  TinyModel, CLI smoke.
