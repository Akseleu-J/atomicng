# Changelog

AtomicNG follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and Semantic Versioning.

## [1.0.1] — 2026-11-13

### Fixed (fidelity review by the method author, issue #1 on the reference repo)
- **c_j counts BLOCKS, not events**: D1 averages a block's events into one update, so the
  counter grows by 1 per writing block (previously `+= n_j` per event). Affects the gate
  input `log(1 + c_j)`. Fixed in `rules/d1.py` and both trainers; the count test now asserts
  the block semantics (1 per block, 2 after two blocks).
- **Undo is LIFO**: documented in `journal.py` and README — exact for the most recent block;
  older blocks whose rows were touched later cannot be reverted exactly.
- Sidecar projection ships with writer checkpoints (`save_sidecar_proj` / `load_sidecar_proj`)
  — published writers are reproducible end-to-end (projection deterministic from seed 17).
- CALIBRATION.md: SmolLM2 results explicitly labeled **external-table setup**; acquisition
  reported as Δ_empty + Δ_donor pairs (forgetting can raise Δ_donor).
- README: explicit independent-implementation statement.

## [1.0.0] — 2026-11-05

First stable release. A golden test freezes the public API.

### Added

- NumPy core for T = T⁰ + Δ, with D1 block averaging, atomic write guards, pre-image undo, full reset, and data-only overlays with SHA-256 and Ed25519 support.
- Plain and delta MLP writers, with BPTT through the D1 update chain on PyTorch CPU, CUDA, and XLA paths.
- Track B native read-out in `native.py` (16 heads × 160 dimensions, followed by value and gated key paths with 10,240 output dimensions).
- A `vllm.general_plugins` entry point, merged-table export for vLLM, and PLE export for llama.cpp.
- Acquisition, distraction, and chain evaluations, with 512-token evidence blocks and a 128-token query prefix.
- Write reports, metrics exporters, standalone HTML monitoring, regression checks, publication checks, and an audit log.
- Documentation, a Colab TPU notebook, and core, PyTorch, build, and publication workflows.

### Recorded validation

- 85 of 85 tests on Windows and 72 of 72 on Linux with CUDA T4 and Python 3.13. The CUDA run verified bit-exact restoration after clearing Δ.
- The 860-step SmolLM2-135M writer run recorded acquisition Δ_empty = +0.0422 on one document.
- The JAX D1 harness recorded 8.4 ms/iteration on `TPU_0`. See [CALIBRATION.md](CALIBRATION.md) for the run conditions and limits.

## [0.9.0rc1] — 2026-11-04

### Added

- MkDocs Material documentation and a GitHub Pages workflow. `scripts/build_docs.py` assembles the site from the repository Markdown.
- `test_api_surface` freezes `__all__`, reports added and removed names, and requires intentional changes to be documented.
- Examples and a release checklist covering PyPI, the Colab TPU run, and vLLM validation.

### Fixed

- Stabilized `test_trainer_d1_chain_is_bptt` with deterministic blocks and the trainer's guard for groups without a gradient path.

## [0.5.0a2] — 2026-11-04

### Added

- The `atomicng_ngram_overlay` general plugin reads `VLLM_ATOMICNG_OVERLAY` and exposes a provider for T⁰+Δ. It returns an empty mapping when no overlay is configured and does not import vLLM.
- `atomicng export-vllm` writes a merged NPZ table and `vllm_config.json` with SHA-256 and provenance. The integration guide documents three paths and includes an upstream issue draft.
- Six bridge tests cover merge behavior, environment configuration, export artifacts, and the entry point.
- README badges, the calibration matrix, and a direct Colab link.

## [0.5.0a1] — 2026-11-04

### Added

- Track B native adapter primitives in `native.py`. `NativeTableSpec` defines eight bigram heads and eight trigram heads of width 160. `native_head_rows` and `native_rows_all_heads` compute row addresses. `NativeReadOut` concatenates rows into ℝ²⁵⁶⁰ and applies value and gated key paths to produce ℝ¹⁰²⁴⁰. The key path uses RMSNorm, a signed square-root transform with a floor, and four-tap causal depthwise convolution with dilation 3.
- Recorded calibration across Windows CPU, Kaggle CPU, CUDA T4 (73 ms/group), Colab 24-core CPU (54 ms/group, preliminary), and TPU. The 860-step CPU run recorded acquisition of +0.0422.

### Fixed

- Skip backpropagation for groups without a gradient path in `WriterTrainer.train_group`. The BPTT test now uses deterministic blocks with repeated rows.
- Convert NumPy patch values to tensors in `test_gather_uses_memory`.

## [0.4.0a3] — 2026-11-02

### Fixed

- Place `SidecarHandle.forward_hidden` token IDs on the model device, move CUDA tensors to CPU before calling `.numpy()` in `SidecarTable.read()`, and place injected reads on the projection device.
- Keep full precision in acquisition results so comparison with NLL differences avoids double rounding.
- Make `test_publish_gate` use a self-contained YAML file in its temporary directory.

### Recorded validation

- The initial Kaggle T4 run recorded BPTT loss 5.55 → 4.05 at 73 ms/group and passed 70 of 72 tests. This release addressed both failures.
- Kaggle CPU recorded 860 SmolLM2-135M writer steps in 7.8 hours, loss 3.36 → 3.12 (min 1.606), acquisition Δ_empty = +0.0422, and 71 of 72 tests passing.
- The TPU run exercised on-device XLA BPTT. The session was canceled after about two hours, with S0–S4 artifacts collected.

## [0.4.0a2] — 2026-11-01

### Added

- Device support for `HFWriterTrainer(..., device=...)` and `SidecarTable.to(device)`. The trainer places the model, writer, sidecar table, and training tensors on the selected device.

### Fixed

- Create `SidecarHandle.nll` indices on the input device for CUDA compatibility.

## [0.4.0a1] — 2026-10-31

### Added

- Source and wheel builds, Twine checks, and a smoke check against a wheel installed in a clean environment. The CI build job checks package artifacts.
- A quickstart notebook, a Python 3.12 slim Docker image with a build-time smoke check, and a Docker ignore file.
- Write-time plain encoding via `handle.write(..., memory_aware=False)`. The writer sees zero Δ_j and c_j at the gate input. Memory writes still apply normally and delta encoding remains the default.
- `atomicng monitor` produces standalone HTML with patch state, a row norm histogram, write metrics, and the verification result.
- `atomicng export-ple` exports T⁰+Δ in the binary `DNGPLE01` format with a JSON header, float32 rows, and SHA-256 integrity checks. The loader validates magic bytes, integrity, and the table specification.
- The test count rose from 66 to 72.

### Fixed

- Resolve a circular import by reading the version lazily in `monitor_html.py`.

## [0.3.2a1] — 2026-10-31

### Added

- `paper.py` centralizes paper constants, including 512/128 blocks, groups of 12, AdamW at 3e-4 with clip 1 and zero weight decay, ρ=0.01, 16 heads of width 160, approximately 4,075,681 writer parameters, and reference acquisition and retention figures.
- Twelve conformance tests cover D1 equations, gate inputs, Equation 7 features, writer parameter count (±5% on the native configuration), ρ calibration, trainer and evidence defaults, and α=1. `FIDELITY.md` documents conformance, deviations, and the clean-room policy.
- `calibrate_to_table(t0)` sets the writer target scale using ρ=0.01 and the base-table RMS.
- Detached Ed25519 signatures for overlays and writer checkpoints through `generate_keypair`, `sign_file`, and `verify_file`. Integrity manifests detect corruption and signatures detect substitution.
- The test count rose from 54 to 66.

### Changed

- `WriterMLP` uses linear targets by default. Set `bound_targets=True` to enable tanh bounding. Writer artifacts preserve this setting across save and load.

## [0.3.1a1] — 2026-10-31

### Added

- `HFWriterTrainer` trains a writer on a real Hugging Face model. The differentiable Δ injection sends gradients through the frozen model and projection to the writer. Loss uses cross-entropy over the block continuation. Groups without a gradient path are skipped.
- `SidecarHandle.nll` reports mean NLL in nats per token under the current memory.
- `scripts/train_writer.py` builds blocks, trains with BPTT, saves a safetensors checkpoint with model ID and corpus SHA-256, and evaluates acquisition on held-out documents.
- The first AtomicNG writer checkpoint trained on SmolLM2-135M used a public-domain Gutenberg corpus and a small CPU configuration.
- 54 tests, including real-model training, finite gradients, frozen model weights, checkpoint round-trip, and NLL changes after writing.

### Fixed

- Removed a `no_grad` scope that disconnected the writer from BPTT. Removed redundant dimension expansion of `counts_t` and Ĥ, whose hook already returns (B, T, Hd).

## [0.3.0a1] — 2026-10-31

### Added

- Backend-independent acquisition (Δ_empty and Δ_donor), distraction (retention@k), and chain (neutral drift, recall, and stability) protocols. Each takes a patch snapshot and restores it after evaluation.
- `TinyModel.token_logits` and `TinyModel.nll`.
- Data-only safetensors writer artifacts with version and architecture validation, plus `writer_fingerprint` for SHA-256 provenance.
- `atomicng eval` with quick, acquisition, distraction, and chain suites and optional JSON output.
- The test count rose from 40 to 51. Coverage includes protocol state restoration, artifact round-trip and fingerprints, real Llama sidecar integration with bit-exact reset, and a positive Δ_empty test using a trained writer on the small PyTorch backend.

### Changed

- The PyTorch extra now includes `safetensors>=0.4`.

## [0.2.0a2] — 2026-10-31

### Changed

- Renamed the prerelease project from `dng` to `atomicng`, including the import, CLI, overlay tag, and specification tag. References to the paper retain its original method name, Delta N-gram.

## [0.2.0a1] — 2026-10-31

### Added

- `WritePolicies` with atomic row and write budgets and soft and hard norm caps. D1 calculates and validates proposed rows before applying them.
- `build_blocks` with the paper's 512-token blocks and 128-token query prefix, plus `ByteTokenizer`.
- A metrics registry with JSONL, TensorBoard, and W&B exporters for gates, norms, hot rows, row reuse, and events per second.
- `atomicng bench` measures write throughput, read overhead, and verification time. `check_regression` supports CI.
- `atomicng publish` checks verification, drift on neutral probes, and hot rows.
- `WriterMLP`, `TorchTinyModel`, and `WriterTrainer`, with BPTT through D1, one backward pass per group, AdamW at 3e-4, clip 1, and a gradient stop at Ĥ. The Hugging Face sidecar injects reads before layer L and captures Ĥ.
- SHA-256 overlay manifests, D1 merging, provenance, YAML configuration, typed package metadata, CI, and Ruff checks.

### Changed

- Verification and publication use probes that avoid patched rows.
- D1 limits moved from `D1Rule` to `WritePolicies`.

## [0.1.0] — 2026-10-31

### Added

- NumPy core with `TableSpec`, sparse `MemoryPatch`, lazy decay, snapshots, overlay round-trip, D1 block averaging, pre-image undo, audit logs, bit-exact deletion verification, `TinyModel`, and CLI smoke checks.
- Property tests for convexity, boundedness, and within-block order invariance.
