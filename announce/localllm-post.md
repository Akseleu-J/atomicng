# r/LocalLLM — пост (практичная аудитория: локальные модели на своём железе)

**Заголовок:**
Your local model can now *remember* — forward-only memory writes, no retraining, instant rollback (open source)

**Тип:** Link → https://github.com/Akseleu-J/atomicng

**Текст:**

Ever wanted your local LLM to actually *learn* a fact without fine-tuning — and be able to undo it completely?

**AtomicNG** (Apache-2.0, just hit 1.0.0 on PyPI) implements a paper called "Delta N-gram": your model's n-gram sidecar table stays frozen, and new knowledge goes into a small sparse patch Δ. Writes are forward passes only — no gradients, no weight edits. **Clear the patch and the model is byte-identical to base.** No "unlearning" gymnastics — deletion is structural.

Quickstart (CPU, no GPU, pure NumPy core):

```python
import atomicng

model = atomicng.TinyModel()
atomicng.write(model, "New fact about kangaroos.")
atomicng.verify_deletion(model)   # clear Δ ⇒ byte-identical to base — guaranteed
```

**What's inside:**
- A trained writer checkpoint ships with the Release (860 training steps on SmolLM2-135M; measured effect: writing a doc *lowers* its NLL vs empty memory)
- Per-tenant memory: overlays are small data-only files — merge them, revert them, sign them
- vLLM support via `vllm.general_plugins` + merged-table export
- Built-in eval: does the memory write actually help? (paper protocols: acquisition, distraction retention, chain drift)

**Calibrated on real hardware:** Windows/Linux CPU, CUDA T4 (73 ms/group BPTT stand), TPU chip via JAX (8.4 ms/iter). Full matrix: CALIBRATION.md in the repo.

**Honest limits:** acquisition needs a few hundred training steps of the writer; the memory lives in the n-gram table (facts/associations, not long reasoning); vLLM engine-graph hooking is planned upstream.

GitHub: https://github.com/Akseleu-J/atomicng
Colab TPU (one click): https://colab.research.google.com/github/Akseleu-J/atomicng/main/notebooks/tpu_calibration.ipynb

Questions welcome — I'll be in the comments.
