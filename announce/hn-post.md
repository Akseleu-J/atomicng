# HN-пост (не Show HN — содержательный, «по сути»)

## Заголовок (выбери один)

1. `Forward-only memory writes for LLMs: clearing the patch restores the model byte-for-byte`
2. `A sparse patch over a frozen n-gram table: forward-only knowledge writes for LLMs`
3. `Writing new facts into an LLM's n-gram table without touching its weights`

Рекомендую #1 — самое конкретное утверждение, легко проверяется (это и есть демонстрация).

## Тип сабмита

**Text** (текст вместо URL — репо-ссылка идёт первым абзацем тела; так HN показывает текст под заголовком).

## Тело поста

I implemented a paper called "Delta N-gram" (arXiv:2610.31125) as an Apache-2.0 framework called AtomicNG: https://github.com/Akseleu-J/atomicng

The idea: some recent LLMs have an n-gram sidecar table that the model reads alongside its weights. The paper shows this table can be treated as a writable memory. The pretrained table T⁰ stays frozen; new facts go into a sparse patch Δ through forward passes only — a ~4M-param writer turns text into row targets and gates, and a block-averaged DeltaNet-style rule folds them into Δ. No gradients, no weight edits.

The property I care most about: clearing Δ restores the base model byte-for-byte. Deleting knowledge is structural, not approximated. Also, overlays are small data-only files — they can be merged, reverted exactly via a pre-image journal, and signed.

What I verified with real runs (not claims):

- 85/85 tests on Windows-CPU, 72/72 on Linux+CUDA T4 (Python 3.13)
- Bit-exact restore of outputs after clearing Δ, on CPU without flags and on CUDA with deterministic algorithms
- A writer trained for 860 BPTT steps on SmolLM2-135M (sidecar 4096×16, Gutenberg corpus), then measured on held-out paragraphs: writing a doc changes its own NLL by ≈0 (max +0.0033) — sanity-check territory at this scale
- Writing a doc makes its paraphrase worse: −0.09 … −2.05 nats (mean −0.89). Mechanism: rows written for the doc's bigrams fire inside the paraphrase carrying deltas derived from a different context — hash-collision interference
- Unrelated held-out text after 8 writes: exactly 0.0 delta — locality holds
- D1 math (read → writer → write) runs on a TPU chip in 8.4 ms/iter via JAX

Honest limitations: the paraphrase damage above is the real limit at this training scale — the paper's writer trains on 100M tokens and that regime resolves it; multi-seed retraining is queued. The vLLM engine-graph hook is pending upstream; today you export a merged table and attach it at load time. The framework is an independent implementation — the author of the method reviewed it (issue on the reference repo) and the fixes from that review are in 1.0.1.

Negative control for the deletion check: the verify routine now compares against a reference model (same architecture/seed, empty memory) on probes that target patch rows. A simulated bug where clear() removes only half the affected rows is caught by this comparison — self-comparison alone passes it, which was a fair criticism.

Full calibration matrix: https://github.com/Akseleu-J/atomicng/blob/main/CALIBRATION.md
Docs: https://akseleu-j.github.io/atomicng/
