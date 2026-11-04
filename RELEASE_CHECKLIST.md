# RELEASE_CHECKLIST — v1.0.0

Статус: **0.9.0rc1** — v1.0.0 после закрытия трёх пунктов ⏳.

## Готово (не трогать)

- [x] 83/83 тестов (Windows) + 72/72 (Linux+CUDA T4, Python 3.13) — верификация atomicng-gpu-verify-1
- [x] FIDELITY-конформность статье (conformance-тесты), CALIBRATION.md — живая матрица
- [x] Побитовый откат: CPU без флагов, CUDA с deterministic-режимом
- [x] Пакет: build + twine PASSED; entry-point vllm.general_plugins
- [x] GitHub публичный: README с бейджами, FIDELITY, CALIBRATION, docs/vllm.md, Colab-кнопка
- [x] GitHub Release v0.5.0a2: wheel + обученный writer (acquisition +0.0422)
- [x] Docs-сайт: mkdocs Material + Pages workflow (docs.yml)
- [x] Первый настоящий acquisition: 860 шагов, Δ_empty +0.0422

## ⏳ До v1.0.0 (владелец)

1. **PyPI**: `twine upload dist/atomicng-0.9.0rc1*` → затем `pip install atomicng` работает.
2. **Colab TPU прогон**: https://colab.research.google.com/github/Akseleu-J/atomicng/main/notebooks/tpu_calibration.ipynb
   (TPU runtime → Run all; отчёты /content/results → скинуть в evidence).
3. **vLLM реальный стенд** (Linux): путь 1 из docs/vllm.md (merged table) на модели с заводской
   таблицей → запись в CALIBRATION.md.
4. **GitHub Pages**: Settings → Pages → Source: `gh-pages` ветка (создастся workflow docs.yml после
   следующего push в main).
5. **Опционально**: upstream issue в vllm-project/vllm (черновик в docs/vllm.md).

## Команды релиза

```sh
# PyPI
python -m twine upload dist/atomicng-0.9.0rc1*
# после вливания всех фиксoв — финальный тег
git tag v1.0.0 && git push origin v1.0.0
gh release create v1.0.0 --title "AtomicNG 1.0.0" --generate-notes
```
