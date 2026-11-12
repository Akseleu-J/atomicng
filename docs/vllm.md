# AtomicNG and vLLM integration

AtomicNG writes to a sparse overlay `Δ` while keeping the model weights and base n-gram table fixed. The effective table is `T = T⁰ + Δ`. This guide describes two integration paths and an upstream proposal.

## 1. Export a merged table

Export the full table and load it before inference:

```sh
pip install atomicng
atomicng write --evidence docs.txt --overlay mem.atomicng.npz
atomicng export-vllm --overlay mem.atomicng.npz --base-table t0.npz \
    --model-id Qwen3.8-Flash-Next --out ./vllm-artifacts
# Output: vllm-artifacts/ngram_table_merged.npz and vllm_config.json
```

The configuration records the merged table path, table dimensions and heads, SHA-256, and provenance. The runner loads the merged table before inference, so it does not apply the overlay on each read.

## 2. Use the general plugin

AtomicNG registers a `vllm.general_plugins` entry point. To supply an overlay when starting the server:

```sh
pip install atomicng
export VLLM_ATOMICNG_OVERLAY=/path/mem.atomicng.npz
vllm serve <model> ...
```

With no overlay configured, the plugin returns an empty dictionary. With an overlay, it provides `atomicng_ngram_overlay(base_table, spec)`, which returns the merged table and its `MemoryPatch`.

## 3. Upstream Engram proposal

For models with a built-in n-gram table, the proposal adds `--ngram-overlay path` to the existing Engram/PLE load path.

## Draft issue for vllm-project/vllm

**Title:** Feature: overlay support for n-gram/Engram sidecar tables (forward-only memory writes)

**Body:**
AtomicNG (Apache-2.0, arXiv:2610.31125) implements forward-only continual writing into an
n-gram sidecar table: pretrained table T⁰ stays frozen, new knowledge accumulates in a sparse
patch Δ with block-averaged DeltaNet-style updates (convex combination ⇒ bounded norms), and
clearing Δ restores the base model bit-exactly. Writes are forward-only (no gradients).

Proposal: allow serving with an n-gram overlay — `--ngram-overlay path` — that adds Δ on top
of the factory table at load time. Artifacts are data-only safetensors/npz with sha256
manifests and provenance; a general-plugin entry point (`vllm.general_plugins` →
`atomicng_ngram_overlay`) already ships in `pip install atomicng` and can supply the merged
table (T⁰+Δ) to the engine. Calibration on SmolLM2-135M sidecar and a Tesla T4/Linux
verification (72/72 tests, bit-exact restore) are published in the repo.

Use cases: continual knowledge updates without retraining; per-tenant memory overlays
(merge/revert via pre-image journal); A/B of knowledge versions by swapping overlays.

## Compatibility

- The merged table uses little-endian float32 with shape `(N, d)`, matching the overlay specification.
- The plugin does not import vLLM. The plugin tests can run without it.
- See [CALIBRATION.md](../CALIBRATION.md) for the status of live vLLM calibration on Linux.
