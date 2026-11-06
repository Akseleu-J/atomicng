# AtomicNG: update your LLM's n-gram memory forward-only — no weight edits, bit-exact rollback

**TL;DR:** I implemented [Delta N-gram (arXiv:2610.31125)](https://arxiv.org/abs/2610.31125) as a pip-installable framework — **AtomicNG**. It writes new knowledge into a model's n-gram sidecar table with forward passes only (no gradients), keeps the pretrained table frozen, and clearing the patch restores the base model **bit-exactly**. Apache-2.0, core is pure NumPy.

GitHub: https://github.com/Akseleu-J/atomicng
PyPI: `pip install atomicng` (1.0.0)
Colab TPU calibration (one click): https://colab.research.google.com/github/Akseleu-J/atomicng/main/notebooks/tpu_calibration.ipynb

## What it does

```
T = T⁰ + Δ        # pretrained table frozen; new facts → sparse patch Δ
```

- A tiny writer (~4M params) turns text into row targets + gates; a block-averaged DeltaNet-style rule folds them into Δ (convex combination ⇒ norms stay bounded — no explosion over thousands of writes).
- **Deletion is structural**: clear Δ ⇒ the model is byte-identical to base. Verified on CPU and CUDA (with deterministic algorithms).
- Writes are **forward-only** — no optimizer state, no weight surgery, no rollback complexity.
- Overlays are data-only artifacts (npz/safetensors + sha256 manifest + provenance), mergeable and revertible (pre-image journal).

## Real numbers (all reproducible, see CALIBRATION.md)

| What | Result |
|---|---|
| Tests | 85/85 (Windows-CPU), 72/72 (Linux+CUDA T4, Python 3.13) |
| Trained writer | 860 BPTT steps on SmolLM2-135M (Gutenberg corpus) → **acquisition Δ_empty = +0.0422** |
| Bit-exact restore after clearing Δ | yes — on CPU without flags, on CUDA with deterministic mode |
| D1 math on TPU chip (JAX) | 8.4 ms/iter |
| BPTT stand on T4 | 73 ms/group (~2000× faster than my laptop CPU) |
| vLLM | bridge via `vllm.general_plugins` + `atomicng export-vllm`; engine verified on T4 |

## The trained writer

The Release ships `writer_night.safetensors` — a writer after 860 BPTT steps (trained on the Gutenberg Sherlock corpus, sidecar table 4096×16 attached to SmolLM2-135M). It's the first published checkpoint with a measured positive acquisition (memory write lowers NLL of the written doc vs empty memory). Load it:

```python
from atomicng.writers.io import load_writer
writer, config = load_writer("writer_night.safetensors")
```

## Honest limitations (read before trying)

- The engine-graph hook for vLLM (loading the overlay inside the running engine) is Track B — today you consume the merged table via export, the upstream Engram-path PR is planned. Draft issue text is in `docs/vllm.md`.
- BPTT training on the TPU chip itself needs the torch/torch_xla pair restarted (ABI mismatch on Colab images); D1 math runs on the chip via JAX already.
- The writer needs a few hundred training steps before acquisition shows up — don't expect magic from an untrained one.
- This implements the paper's method (credit to Hao Wu) — I did not invent it; the framework, tests, calibration and the training pipeline are mine.

## Quickstart (CPU, no GPU needed)

```python
import atomicng

model = atomicng.TinyModel()
atomicng.write(model, "New fact about kangaroo navigation.")
atomicng.verify_deletion(model)   # clear Δ ⇒ byte-identical to base — guaranteed
```

Docs: https://akseleu-j.github.io/atomicng/ · Calibration: [CALIBRATION.md](https://github.com/Akseleu-J/atomicng/blob/main/CALIBRATION.md) · Paper-fidelity tests: `pytest tests/test_fidelity.py`

Happy to answer questions — I'll be in the comments for the next 24h.
