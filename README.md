# AtomicNG

[![CI](https://github.com/Akseleu-J/atomicng/actions/workflows/ci.yml/badge.svg)](https://github.com/Akseleu-J/atomicng/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Colab TPU](https://img.shields.io/badge/Colab-TPU%20calibration-F9AB00?logo=google-colab)](https://colab.research.google.com/github/Akseleu-J/atomicng/main/notebooks/tpu_calibration.ipynb)
[![vLLM](https://img.shields.io/badge/vLLM-general__plugin-red)](docs/vllm.md)

**AtomicNG** implements atomic, forward-only continual writes to a language model's n-gram memory. It follows the method described in [arXiv:2610.31125](https://arxiv.org/abs/2610.31125) (Hao Wu) — **an independent implementation**, not affiliated with the author; reference code lives at `af-74413592/delta-ngram`. The pretrained n-gram hash table `T⁰` stays frozen, and writes accumulate in a sparse patch `Δ` such that `T = T⁰ + Δ`. Write guards enforce budgets and caps before the patch changes. Clearing `Δ` restores the base model bit for bit.

The Apache-2.0 core uses NumPy. The optional PyTorch layer provides writers and Hugging Face integration. See the [vLLM integration guide](docs/vllm.md) for the general plugin.

[CPU, CUDA, and TPU calibration results](CALIBRATION.md) · [Run TPU calibration in Colab](https://colab.research.google.com/github/Akseleu-J/atomicng/main/notebooks/tpu_calibration.ipynb)

## Try the CPU demo

```sh
python -m pip install atomicng==1.0.0
```

Run the following with Python. It uses the deterministic NumPy test backend and needs no model download or GPU. The matching probe and eight neutral probes are measured separately.

```python
import atomicng

model = atomicng.TinyModel()
writer = atomicng.HashWriter(model.spec)
text = "A field note about volcanic soil chemistry."
query = writer.tokenize(text)

before = model.logits(query).tobytes()
report = atomicng.write(model, text, writer=writer)
written = model.logits(query).tobytes()
assert report.n_rows > 0 and written != before
print(f"Write: {report.n_rows} rows touched. Matching probe changed.")

neutral = model.neutral_token_seqs(8)
neutral_written = model.run_seqs(neutral).tobytes()
atomicng.clear_memory(model)

assert model.logits(query).tobytes() == before
assert model.run_seqs(neutral).tobytes() == neutral_written
print("Clear: matching probe restored byte for byte.")
print("Control: eight neutral probes unchanged.")
```

The example measures probe outputs. For writer training on a Hugging Face model, see the [calibration results](CALIBRATION.md).


## Installation

```sh
pip install -e .            # Core dependencies: NumPy and PyYAML
pip install -e ".[torch]"   # Writers and Hugging Face backend
pip install -e ".[dev]"     # Development and test dependencies
```

## Quickstart

```python
import atomicng

model = atomicng.TinyModel()  # Local test backend
policies = atomicng.WritePolicies(row_budget=100_000, hot_row_norm_cap=1.0)
metrics, audit, undo = atomicng.MetricsRegistry(), atomicng.AuditLog("audit.jsonl"), atomicng.UndoJournal()

report = atomicng.write(
    model, "A document containing a new fact...",
    policies=policies, metrics=metrics, audit=audit, undo=undo, source="doc-1",
)
model.memory.save_pretrained("out.atomicng.npz")  # Data-only overlay and SHA-256 manifest
atomicng.publish_check(model)  # Checks verification, drift, and hot rows
```

The Hugging Face backend uses `atomicng.backends.hf.attach_sidecar`. The CLI reads YAML configurations from [configs/](configs/).

```sh
atomicng write --evidence docs.txt --overlay mem.atomicng.npz --config configs/write_default.yaml
atomicng stats mem.atomicng.npz          # Rows, norms, reuse, and provenance
atomicng verify mem.atomicng.npz         # Check the SHA-256 integrity manifest
atomicng merge --base a.atomicng.npz --other b.atomicng.npz --out c.atomicng.npz
atomicng publish mem.atomicng.npz --config configs/publish_strict.yaml
atomicng bench --save baseline.json && atomicng bench --docs 200 --check baseline.json
atomicng eval --evidence examples/eval_docs.txt --suite quick --json report.json
atomicng monitor mem.atomicng.npz --metrics m.jsonl            # Standalone HTML report
atomicng export-ple --overlay mem.atomicng.npz --out table.ple  # Table for llama.cpp
atomicng smoke
```

## Write safety and integrity

| Mechanism | Behavior |
|---|---|
| Immutable base table | Writes affect only the sparse `Δ` store. |
| Atomic guards | Row and write budgets and hard caps are checked before the patch changes. |
| Write report | Each write reports rows, gates, norms, and hot-row flags. |
| Exact undo | `atomicng.revert_last` restores the last block from its pre-image journal. **LIFO only**: exact for the most recent block; older blocks whose rows were touched later cannot be reverted exactly (D1 depends on current row state). |
| Full reset | `model.memory.clear()` removes the full patch. |
| Overlay integrity | The format stores data only (NPZ and JSON) with a SHA-256 manifest. |
| Publication checks | Verification must pass, and drift and hot-row counts must stay within their limits. |
| Audit | An append-only JSONL log records writes and merges with checksums. |

## Monitoring

`MetricsRegistry` supports JSONL and optional TensorBoard and Weights & Biases exporters. It records gates, patch norms, hot rows, row reuse, and events per second, both per write and in aggregate.

## Architecture

```text
src/atomicng/
├── specs.py        # Versioned table specifications and validation
├── memory.py       # Sparse patch, lazy decay, merge, and overlay integrity
├── guards.py       # Atomic write budgets and caps
├── rules/d1.py     # Two-phase D1 block averaging
├── evidence.py     # Document blocks (512/128) and ByteTokenizer
├── metrics.py      # Metrics registry and exporters
├── bench.py        # Throughput, overhead, and regression checks
├── publishing.py   # Verification, drift, and hot-row checks
├── journal.py      # Audit log and pre-image undo
├── verify.py       # Bit-exact deletion checks on neutral probes
├── backends/       # Tiny CPU backend and Hugging Face sidecar hooks
└── writers/        # Plain/delta WriterMLP and BPTT through the D1 update chain
```

The core runs without PyTorch. Writer training and the Hugging Face sidecar require the `torch` extra. Full JAX model attachment is roadmap work.

## Tests

```sh
python tests/run_all.py        # Or run pytest -q
ruff check src tests
```

The suite covers D1, gates, Eq. 7, writer parameters, and ρ calibration, along with plain encoding, safetensors writer artifacts, Ed25519 signatures, HTML monitoring, PLE export, and the PyTorch layer. Hugging Face integration tests exercise attaching, writing to, and clearing the sidecar on a tiny random Llama model. See [FIDELITY.md](FIDELITY.md) and `pytest tests/test_fidelity.py`.

### Use the pretrained SmolLM2-135M Writer

The [860-step Writer checkpoint](https://github.com/Akseleu-J/atomicng/releases/download/v0.5.0a2/writer_night.safetensors) is publicly downloadable. Run the [CPU example](examples/pretrained_writer.py) from this checkout to download and verify the checkpoint, load the pinned SmolLM2-135M base model, and check a forward-only write and bit-exact rollback.

```sh
python -m pip install "atomicng[torch]==1.0.0" transformers
python examples/pretrained_writer.py
```

The checkpoint contains Writer weights. The example initializes a reproducible sidecar projection because the original run did not save its projection. The example reports the measured NLL on one input alongside the write and rollback checks. See [the calibration record](CALIBRATION.md) for the original training results.

### Train a writer on your own model

```sh
python scripts/train_writer.py --model HuggingFaceTB/SmolLM2-135M \
    --corpus corpus.txt --out runs/demo --steps 30
```

The model stays frozen while training updates the writer (approximately 50,000 to 4 million parameters). The CPU configuration uses block length/stride 96/24. The GPU configuration uses 512/128, as in the paper. The example writes `runs/demo/writer.safetensors` with provenance and `runs/demo/eval.json` with per-document `Δ_empty`.

### Release checks

The build job creates a source distribution and wheel, and checks them with Twine. The core CI job runs `atomicng smoke`; release validation also recorded a smoke pass against a wheel installed in a clean environment. The [quickstart notebook](notebooks/QUICKSTART.ipynb) provides runnable examples. The Docker image uses `python:3.12-slim` and runs the smoke check during the build.

See the [changelog](CHANGELOG.md) for release history and the [release checklist](RELEASE_CHECKLIST.md) for publication steps.
