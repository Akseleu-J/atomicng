# AtomicNG calibration

These figures come from recorded AtomicNG 0.4.0a3 and 0.5.0a1 runs. The original evidence references point to the accompanying `kaggle-evidence/` directory.

## Environment matrix

| Environment | Python | PyTorch | Cores or device | pytest | BPTT harness (four-block group) | Writer training |
|---|---|---|---|---|---|---|
| Windows 10, local CPU | 3.12 | 2.x+cpu | 8 cores | **72/72** | ~50 s/group | 40 steps in ~100 min |
| Kaggle Linux, CPU | 3.13 | 2.11+cpu | 4 cores | **71/72** ¹ | S2 failed on a missing `os` import, later fixed | 860 steps in 7.8 h |
| Kaggle Linux, CUDA | 3.13 | 2.11+cu128 | Tesla T4 | **72/72** | **73 ms/group** (reported as ~2,000× CPU in that run) | Stopped before training because of device issues fixed in 0.4.0a3 |
| Colab Linux, CPU | 3.13 | 2.14+cpu | 24 cores | Not run | 54 ms/group ² | 157 steps in 45 min, minimum loss 1.6025 |
| Colab TPU | 3.13 | — | `TpuDevice(id=0)` | — | JAX detected the TPU. See the TPU notes below for the XLA path. | — |

¹ The sole failure was `test_publish_gate`, which depended on the repository's `configs/` path. The fix uses self-contained YAML and passed verification on Linux with CUDA.

² Preliminary. The first run lacked `torch_xla` at S0, but recorded an S3 report after the notebook cells were run out of order.

## Linux CUDA verification for 0.4.0a3

- The Tesla T4 run on Python 3.13 passed **72 of 72 tests**. The acquisition check avoided rounding and the publication gate used self-contained YAML.
- The CUDA path exercised attach, write, and NLL evaluation. Clearing `Δ` restored outputs bit for bit (`max_diff = 0.0`) with deterministic CUDA mode and `CUBLAS_WORKSPACE_CONFIG`.

## SmolLM2-135M writer training

The sidecar used a 4096×16 table and seed 17.

| Run | Steps | Loss | Acquisition Δ_empty |
|---|---|---|---|
| Kaggle CPU, 7.8 h | 860 | 3.36 → 3.12 (min 1.606) | **+0.0422** on document 1, 0 on the others |
| Colab TPU VM, 45 min (run 1) | 157 | 3.36 → ~2.7–3.4 (min 1.6025) | −0.0272 to 0 |
| Colab TPU VM, 45 min (run 2) | 156 | 3.35 → ~2.4–3.4 (min 1.6116) | −0.0173 to +0.0147 (mean +0.0024) |

These runs exercised the end-to-end training path. The longest run showed a positive acquisition signal on one document. Reproducing the paper's 100-million-token training regime remains separate work.

## TPU results

- On Kaggle TPU VM, the XLA BPTT chain reached S3 on device. The session was canceled after two hours.
- On Colab v5e with Python 3.13 and PyTorch 2.14.1, JAX 0.11.2 detected the TPU. The PyPI `torch_xla` build had an ABI mismatch with the preinstalled PyTorch (`_XLAC.so: undefined symbol`). The notebook includes a cell to install a matching pair and restart the runtime.
- On 5 November, the owner ran S3b on `TPU_0` with JAX. The D1 read, writer, and write loop took **8.4 ms/iteration**. The maximum was 0.156 s, including first-iteration JIT compilation. Loss stayed finite across 20 iterations. The recorded report is `kaggle-evidence/colab-tpu/results/S3b_accel_jax.json`.
- The notebook's XLA path is used for on-device BPTT training after restarting with the matching packages.

## vLLM on Kaggle Linux and Tesla T4

- `export_for_vllm` generated `ngram_table_merged.npz` and `vllm_config.json` with SHA-256 and provenance on the target platform.
- vLLM generated text with `facebook/opt-125m` at **21.4 output tokens/s** on a T4. AtomicNG artifacts were present in the same environment. An isolated `uv` virtual environment avoided the conflict with Kaggle's preinstalled PyTorch. See `kaggle-night-vllm/verify_vllm3.py`.
- Loading the merged table into the engine graph remains Track B upstream work. See the [vLLM integration guide](docs/vllm.md).

## Known limits

- Bit-exact CUDA restoration requires deterministic mode. CPU restoration does not require those flags.
- The XLA path keeps bigram hashing on the CPU behind a gradient stop and places the tensors on the TPU.
- The vLLM reader has yet to be calibrated.
