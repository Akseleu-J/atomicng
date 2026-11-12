# AtomicNG examples

| Example | What it covers |
|---|---|
| [Quickstart notebook](../notebooks/QUICKSTART.ipynb) | Write, check guards, verify deletion, run evaluations, and check publication on CPU. |
| [TPU calibration notebook](../notebooks/tpu_calibration.ipynb) | Run TPU calibration in [Colab](https://colab.research.google.com/github/Akseleu-J/atomicng/main/notebooks/tpu_calibration.ipynb). |
| [Evaluation corpus](eval_docs.txt) and [report](eval_report.json) | Acquisition, distraction, and chain examples on 20 documents. |
| [Writer training script](../scripts/train_writer.py) | Train a writer against a Hugging Face model such as SmolLM2-135M. |
| [Trained writer release](https://github.com/Akseleu-J/atomicng/releases/tag/v0.5.0a2) | The 860-step checkpoint associated with the +0.0422 acquisition result. |

## Write to memory

```python
import atomicng

model = atomicng.TinyModel()
atomicng.write(model, "Your first fact in n-gram memory.")
atomicng.verify_deletion(model)  # Verify bit-exact rollback.
```
