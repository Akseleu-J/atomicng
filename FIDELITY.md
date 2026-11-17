# AtomicNG conformance to the paper

AtomicNG tracks conformance to Hao Wu's 2026 paper, *Delta N-gram: Forward-Only Continual Writing into a Language Model's Native N-gram Memory* (arXiv:2610.31125). [paper.py](src/atomicng/paper.py) holds the reference constants. The tests named below check the corresponding implementation where applicable.

## Clean-room implementation

The project records that the upstream repository `af-74413592/delta-ngram` had no license when AtomicNG was implemented. The clean-room implementation used the paper and the upstream public README for documented behavior, CLI concepts, and smoke configuration. AtomicNG uses its own writer checkpoint and overlay formats.

## Conformance matrix

**✓** matches the paper and has a test where applicable. **≈** has a documented deviation. **✗** marks an open gap. **+** marks an extension.

| Item | Paper | AtomicNG | Status |
|---|---|---|---|
| Decomposition T = T⁰ + Δ | T⁰ is frozen and Δ is sparse over touched rows. | `MemoryPatch` stores only touched rows and has no write access to T⁰. | ✓ (`test_core`, verify) |
| D1 write rule | A_j=(1/n)Σβ_a v_a, w_j=(1/n)Σβ_a, Δ⁺=(1−w)Δ+A, **c⁺=c+1 per block** (D1 averages a block's events into one update, so c_j counts writing blocks — confirmed by the author, issue #1 on the reference repo) | `rules/d1.py` applies the rule in two phases with `c_j += 1` per touched row per block. The test compares it directly with the equations. | ✓ (`test_fidelity::d1`, `test_counts_increment_per_block`) |
| Boundedness (Proposition 1) | ‖Δ_j⁺‖ ≤ max(‖Δ_j‖, max‖v_a‖) | Property test over a random chain of 50 blocks. | ✓ (`test_core::bounded`) |
| Within-block order invariance | Block averaging is commutative. | Property test compares checksums after shuffling events. | ✓ (`test_core::order`) |
| Gate | β_a = σ(B_θ[x_a, Δ_j, v_a−Δ_j, log(1+c_j)]) | `WriterMLP.b_head` uses F+2d+1 input features and a sigmoid. | ✓ (`test_fidelity::gate`) |
| Equation 7 features | [Ĥ_p; mean p+1..p+4; block pool; (order, head) embedding], with gradients stopped at Ĥ | `WriterMLP.forward` uses F=3·Hd+emb, detaches Ĥ, and sets `FEATURE_NEXT_WINDOW=4`. | ✓ (`test_fidelity::eq7`) |
| Writer architecture | V_θ and B_θ are two-layer MLPs with hidden size 256 and about 4,075,681 parameters combined. | The same formula is used. The native configuration differs by less than 1%. | ✓ (`test_fidelity::params`) |
| Target scale s_h | Fixed s_h gives an initial target RMS of 0.01·RMS(T⁰_h). | `calibrate_to_table(t0, ρ=0.01)` uses N(0,1) noise because the paper leaves the calibration input distribution unspecified. | ≈ (`test_fidelity::rho`) |
| Linear targets | v = s_h·V_θ(x), without clipping | `bound_targets=False` by default. The sidecar can opt into tanh stabilization. | ✓/≈ (`test_torch::tanh`) |
| Plain and delta encoding | Plain (λ=0) encodes on the T⁰-only model: a **four-stage training schedule** (encode all blocks in one batch → D1 chain block-by-block → all queries forward+backward in one batch → backprop along the write chain) reaches ~3,200 target tokens/s vs ~940 interleaved, loss within 0.001 of sequential. Delta (λ=1) encodes with the memory written by earlier blocks; the delta writer is the plain writer forked at 80M tokens and fine-tuned with λ=1 for the last 20M. | Delta-side BPTT is implemented in both trainers (memory-aware). At write time, `handle.write(..., memory_aware=False)` supplies zero Δ and c_j (plain encoding). The **batched plain training schedule is the open gap** — the larger gain is in training speed, not write-time. | ≈ (`test_v04::plain`) |
| Training blocks and query prefix | 512 tokens per block and 128 query tokens | `build_blocks` defaults to 512/128. Small CPU runs explicitly request shorter blocks. | ✓/≈ (`test_fidelity::evidence`) |
| Group normalization | Normalize loss once per group of 12 trajectories. | Group normalization is implemented. Group size is configurable and CPU runs use fewer than 12. | ≈ |
| Optimizer | AdamW, LR 3e-4, no weight decay, clip 1 | Both trainers use these defaults. | ✓ (`test_fidelity::hyper`) |
| BPTT through writes | Backpropagate through the D1 chain to θ and stop gradients at Ĥ. | Both trainers store differentiable Δ rows, backpropagate once per group, and skip groups without a gradient path. | ✓ (`test_hf_training`) |
| Empty starting memory | Reset for each trajectory. | Set `self._delta = {}` at the start of each trajectory. | ✓ |
| Predict before writing | Algorithms 1 and 2 predict the block before writing it. | Both trainers follow this order. | ✓ |
| Decay Δ ← αΔ | Primary setting α=1, with 0.99 also evaluated using a block clock | `MemoryPatch(decay)` applies lazy decay by clock tick and defaults to 1.0. | ✓ (`test_fidelity::evidence`) |
| Delete Δ to restore the base | Clearing Δ restores the model. | `verify_deletion` checks neutral probe outputs bit for bit and verifies the restoration checksum. | ✓ (`test_core`) |
| Native 16-head table | 8 bigram heads, 8 trigram heads, d=160, with reads before layer 2 | `TableSpec` supports this layout. The Track A sidecar uses one bigram head. Native model integration is tracked under Track B. | ✗ for the full native path |
| Paper-scale results | Δ_empty 0.068/0.024, retention 54%/70%, 35,000 writes | `paper.py` records the references. Full-scale reproduction remains open. | ✗ |
| Hot-row guard | The paper observes formatting rows and analyzes the failure mode. | `guards` adds norm caps and budgets, with hot-row metrics. | + |
| Publication checks | Outside the paper's scope | `publish_check` checks verification, drift, and hot rows. Artifacts support Ed25519 signatures. | + |
| Monitoring | Outside the paper's scope | `MetricsRegistry` and `atomicng monitor` produce a standalone HTML report with an SVG norm histogram. | + |
| Engine export | Third-party vLLM patch | `export-ple` writes the merged table in binary `DNGPLE01` format with SHA-256 integrity for llama.cpp. Engine-side vLLM integration is Track B work. | + |
| Undo semantics | The paper does not cover undo. | **+** Pre-image journal; **LIFO only** — exact for the most recent block. Reverting an older block after later blocks touched the same rows is not exact (D1 depends on current row state). Documented in `journal.py`. | + |
| Reported acquisition numbers | Δ_empty (0.068) and Δ_donor (0.024) are reported **together**: forgetting can raise Δ_donor by degrading the donor's memory. | `benchmarks.acquisition` always returns both in one report. | ✓ |

## Documented deviations

1. Track A attaches an external table with its own read projection (`SidecarTable.proj`). Track B covers integration with a model's built-in table. **SmolLM2 sidecar results are external-table calibration** (as the paper treats its prototype), not native-table results; the read projection is deterministic from seed 17 and ships with checkpoints (`save_sidecar_proj`).
2. Small CPU runs explicitly use block lengths of 64–96, prefixes of 16–24, fewer than 12 trajectories per group, and a hidden size of 64 instead of 256. The paper defaults remain 512/128/3e-4.
3. Tanh bounding is optional (`bound_targets=True`). The default targets are linear.
4. Write-time plain encoding is implemented. Batched plain-writer training (four-stage schedule, ~3,200 target tokens/s) remains open.
5. Target-scale calibration uses a noise probe because the paper does not specify the calibration input distribution.

## Updating this record

When changing a listed feature, update its row and corresponding conformance test. Record any new deviation and its rationale. Run `pytest tests/test_fidelity.py` when reviewing conformance.
