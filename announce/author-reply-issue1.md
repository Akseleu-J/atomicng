# Ответ автору на issue #1 (вставить как комментарий в af-74413592/delta-ngram#1)

Hi @af-74413592,

Thank you for the review — this is exactly the kind of check the FIDELITY record exists for. All points are addressed in AtomicNG 1.0.1 ([diff](https://github.com/Akseleu-J/atomicng/compare/v1.0.0...v1.0.1), live on PyPI):

**1. Write count.** Fixed. `rules/d1.py` now grows c_j by 1 per writing block (`c_j += 1`), however many events hit the row, in both trainers as well. The count test now asserts the block semantics (1 after one block with 3 events, 2 after a second block) and FIDELITY.md's D1 entry quotes your reasoning (D1 averages a block's events into a single update; the old n_j-based count made gate inputs order-dependent).

**2. Plain writer.** Agreed — the four-stage schedule's larger gain is in training, not write-time. FIDELITY.md now records it as you described (encode all blocks in one batch → D1 chain block-by-block → all queries forward+backward in one batch → backprop along the write chain; ~3,200 vs ~940 target tokens/s; loss within 0.001 of sequential; delta writer = plain forked at 80M, fine-tuned with λ=1 for the last 20M). The batched plain schedule remains the main open gap on our side — it is now explicitly listed as such.

**3. Sidecar labeling + projection.** CALIBRATION.md now labels the SmolLM2 runs as an external-table setup, mirroring the paper's prototype treatment. The projection in our examples is identical to training (same seed, deterministic init), and `save_sidecar_proj`/`load_sidecar_proj` now ship the projection weights alongside writer checkpoints so published artifacts are reproducible end-to-end.

**4. Reported numbers.** `benchmarks.acquisition` already returns Δ_empty and Δ_donor in one report; CALIBRATION.md now states the pairing rule explicitly (forgetting can raise Δ_donor).

**5. Undo.** Documented as LIFO in `journal.py` and the README safety table: exact for the most recent block; reverting older blocks after later writes touched the same rows is not exact. Non-LIFO scenarios are covered by merge/revert semantics instead.

Also noted the follow-up paper — looking forward to it, and happy to test the improved rule against the conformance suite.

Best,
Akseleu
