# Reddit reply to u/amargolin (EN) — paste as a comment in the r/LocalLLM thread

Ran exactly this on the shipped checkpoint a few minutes ago (860-step writer, SmolLM2-135M, fixed seed, held-out Sherlock paragraphs — script is `scripts/eval_checkpoint_nll.py` in the repo, raw numbers in `announce/checkpoint-eval.json`):

**1. Held-out doc, before → after writing that doc:** ≈ 0 (max +0.0033). You're right — it's an acquisition sanity check, not a generalization result. README relabeled accordingly.

**2. Paraphrase of the written doc (before → after writing the original):** negative — per-doc −0.09 … −2.05, mean −0.89. Writing the doc currently *hurts* its paraphrase. The mechanism is hash-collision interference: rows written for the doc's bigrams carry deltas derived from the doc's context, and they fire inside the paraphrase at positions where that delta is wrong. It's now in "honest limits" in the README — you called it, it's the real limit at this training scale.

**3. Unrelated held-out text after 8 writes:** exactly 0.0 (bit-exact) — perfect locality confirmed.

**4. Negative control for verify_deletion:** implemented, and it can fail. The check now compares outputs against a reference model (same architecture and seed, empty memory) on **hit probes** — probes whose bigrams target patch rows (covering set, one probe per row). A simulated bug where clear() removes only half the affected rows: self-comparison passes (the decoration scenario you predicted), reference comparison fails. Test: `tests/test_verify_negative.py`.

Multi-seed retraining is queued (GPU-hours for the writer); will post the spread when it's done.

Numbers + script are in the repo — easy to rerun.
