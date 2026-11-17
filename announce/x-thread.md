# X and Bluesky announcement thread

**1/4 (hook):**
New framework drop: AtomicNG — write new knowledge into your LLM's n-gram memory with
forward passes only. No gradients. Clear the patch → the model is *byte-identical* to base.
Apache-2.0, pip install atomicng 🧵

**2/4 (how it works):**
Method (arXiv:2610.31125): pretrained n-gram table T⁰ stays frozen. A ~4M-param writer turns
text into row targets + gates; a DeltaNet-style averaging rule folds them into a sparse patch Δ.
Convex combination ⇒ bounded norms. Thousands of writes, zero drift explosions.

**3/4 (number):**
Verified live: 85/85 tests on Windows, 72/72 on Linux+CUDA, bit-exact restore on CUDA,
a writer trained 860 steps on SmolLM2-135M with measured acquisition (+0.0422 NLL drop),
and D1 math on a TPU chip in 8.4 ms/iter. Calibration: CALIBRATION.md in the repo.

**4/4 (reference):**
GitHub: github.com/Akseleu-J/atomicng
Installation: pip install atomicng
One-click TPU Colab: colab.research.google.com/github/Akseleu-J/atomicng/main/notebooks/tpu_calibration.ipynb
Docs: akseleu-j.github.io/atomicng
Star if useful ⭐

---

# GitHub Discussions (announcements)

Title:: **🚀 AtomicNG 1.0.0 — first stable release**
Body: A brief version r/LocalLLaMA post + What backend should we integrate next?»
(vLLM Engram hook / llama.cpp PLE / JAX writer / MLX).
