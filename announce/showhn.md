# Show HN: AtomicNG — forward-only memory writes for LLMs (no weight edits, bit-exact rollback)

Ссылка: https://github.com/Akseleu-J/atomicng

Текст (вставлять вSubmit + ссылка на GitHub; комментарий-first пост — короткий):

---

Hi HN! I built AtomicNG — an Apache-2.0 framework that writes new knowledge into a language model's n-gram sidecar table using forward passes only. No gradients, no weight surgery, and clearing the memory patch restores the base model bit-exactly.

It implements "Delta N-gram" (arXiv:2610.31125): a pretrained n-gram table T⁰ stays frozen, new facts accumulate in a sparse patch Δ via a block-averaged DeltaNet-style rule (convex combination, so patch norms stay bounded), and a ~4M-param writer converts text into row targets and gates. The writer itself is trained with BPTT through the chain of writes.

Why this is interesting:
- Deletion is structural, not a best-effort "unlearn": clear Δ ⇒ outputs are byte-identical to the base model (verified on CPU and CUDA with deterministic algorithms).
- Memory becomes a versioned artifact: overlays are data-only files with sha256 manifests, provenance, merge and exact revert.
- The core is pure NumPy — pip install atomicng works without a GPU.

Verified on real runs: 85/85 tests (Windows-CPU), 72/72 (Linux+CUDA T4, Python 3.13), bit-exact restore on CUDA, a writer trained for 860 steps on SmolLM2-135M with measured positive acquisition (memory write lowers the doc's NLL vs empty memory), and D1 math running on a TPU chip in 8.4 ms/iter.

Honest limitations: the engine-graph hook for vLLM is pending upstream (today you export a merged table); BPTT on the TPU chip needs a matched torch/torch_xla pair; acquisition needs a few hundred training steps to show up.

Repo: https://github.com/Akseleu-J/atomicng
Docs: https://akseleu-j.github.io/atomicng/
Colab (one-click, TPU): https://colab.research.google.com/github/Akseleu-J/atomicng/main/notebooks/tpu_calibration.ipynb

Happy to answer technical questions.

---

Заметки: постить вт–чт утром ET; title строго "Show HN: ..."; в первом комментарии — тот же текст. Если попросят "бенчмарки vs LoRA" — честный ответ: это не замена файнтюна, это ортогональный канал памяти (объём, откаты, мультитенантность).
