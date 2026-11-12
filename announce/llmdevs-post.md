# r/LLMDevs — post (dev audience: API, plugins, PyPI))

**Title**
AtomicNG 1.0.0 — forward-only memory writes for LLMs (no weight edits, bit-exact rollback, vLLM plugin)

**Type:** Link → https://github.com/Akseleu-J/atomicng

**Text:**

I implemented "Delta N-gram" ([arXiv:2610.31125](https://arxiv.org/abs/2610.31125)) as a framework — **AtomicNG 1.0.0**, Apache-2.0, `pip install atomicng`. Core is pure NumPy (no ML framework needed for the memory layer); torch only for writers.

**What it does:** new knowledge goes into a model's n-gram sidecar table via forward passes only. The pretrained table T⁰ stays frozen; facts accumulate in a sparse patch Δ through a block-averaged DeltaNet-style rule (convex combination ⇒ norms stay bounded). Clearing Δ restores the base model **byte-identically** — verified on CPU and CUDA.

**For developers:**
- **API surface is frozen** (1.0.0, golden-tested): `attach_memory` → `write` → `verify_deletion`, 5 lines to first write, core works on CPU without a GPU
- **vLLM**: entry-point `vllm.general_plugins` (`VLLM_ATOMICNG_OVERLAY` env) + `atomicng export-vllm` (merged table + config with sha256). Engine-graph hooking = upstream path
- **Writers are pluggable**: plain/delta MLPs (~4M params), trained via BPTT through the chain of writes (`atomicng.writers.training`); checkpoints are safetensors
- **Overlays are artifacts**: data-only, sha256 manifest, ed25519 signing, merge + exact revert via pre-image journal
- **Evaluation built in**: paper protocols (acquisition Δ_empty/Δ_donor, distraction retention@k, chain drift), `atomicng bench` with CI regression gates
- **Paper-fidelity tests**: a conformance suite pins the implementation to the paper's formulas (see FIDELITY.md)

**Verified on real runs:** 85/85 tests (Windows-CPU), 72/72 (Linux+CUDA T4, Python 3.13), trained writer 860 BPTT steps on SmolLM2-135M with measured positive acquisition (Δ_empty = +0.0422), D1 math on a TPU chip in 8.4 ms/iter (JAX), full calibration matrix in CALIBRATION.md.

**Honest limitations:** the engine-graph hook for vLLM is upstream (merged-table export works today); the writer needs a few hundred training steps before acquisition shows up; this is an orthogonal memory channel, not a fine-tune replacement.

Docs: https://akseleu-j.github.io/atomicng/ · Colab (TPU, one-click): https://colab.research.google.com/github/Akseleu-J/atomicng/main/notebooks/tpu_calibration.ipynb

Method credit: Hao Wu (arXiv:2610.31125). Framework, tests, calibration — mine. Happy to answer questions.
