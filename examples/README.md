# Примеры AtomicNG

| Пример | Что показывает |
|---|---|
| [QUICKSTART.ipynb](../notebooks/QUICKSTART.ipynb) | базовый цикл: запись → guards → verify → протоколы → safety-gate (5 минут, CPU) |
| [tpu_calibration.ipynb](../tpu_calibration.ipynb) | TPU-калибровка на Colab: [один клик](https://colab.research.google.com/github/Akseleu-J/atomicng/main/notebooks/tpu_calibration.ipynb) |
| [eval_docs.txt](eval_docs.txt) + [eval_report.json](eval_report.json) | протоколы статьи (acquisition/distraction/chain) на 20 документах |
| [scripts/train_writer.py](../scripts/train_writer.py) | обучение writer'а на своей модели (SmolLM2-135M → T4/TPU) |
| [writer_night.safetensors](https://github.com/Akseleu-J/atomicng/releases/tag/v0.5.0a2) | **обученный writer** (860 шагов, acquisition +0.0422) — из Release |

## Три строки до первой записи

```python
import atomicng
model = atomicng.TinyModel()
atomicng.write(model, "Твой первый факт в n-gram памяти.")
atomicng.verify_deletion(model)   # откат побитовый — всегда
```
