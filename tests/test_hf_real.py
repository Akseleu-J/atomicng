"""Validation of sidecar-attach on a real HF model (tiny-random LlamaForCausalLM)).

Requires network to download ~2MB of scales; without network, test skipped.
This is the first check of backends.hf outside the tab: the real structure
model.model.layers / model.model.norm / config.hidden_size.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _load_model():
    try:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(MODEL_ID)
    except Exception as exc:  # noqa: BLE001 — Network or model cache may be unavailable.
        pytest.skip(f"HF model {MODEL_ID} idle: {exc}")


def test_real_llama_sidecar_write_and_bit_exact_clear(tmp_path):
    import atomicng
    from atomicng.backends.hf import attach_sidecar
    from atomicng.writers.mlp import WriterMLP

    model = _load_model().eval()
    hidden_size = int(model.config.hidden_size)
    spec = atomicng.TableSpec(name="hf-real", n_rows=512, row_width=hidden_size)
    inject_layer = min(1, len(model.model.layers) - 1)
    handle = attach_sidecar(model, spec=spec, inject_layer=inject_layer)

    vocab = int(model.config.vocab_size)
    ids = [7, 13, 29, 41, 53, 61, 7, 13, 29, 41][: min(10, vocab - 1)]
    ids_t = torch.tensor([ids], dtype=torch.long)

    def logits_now():
        with handle.write_session(ids), torch.no_grad():
            return model(input_ids=ids_t).logits

    base = logits_now()  # The sidecar still injects reads from T⁰ when Δ is empty.

    writer = WriterMLP(hidden_size=hidden_size, row_width=hidden_size, hidden=64, target_scale=0.3)
    report = handle.write(ids, writer)
    assert report["n_events"] == len(ids) - 1

    written = logits_now()
    assert not torch.equal(base, written), "The recording did not affect the release of this model."

    overlay = tmp_path / "real.atomicng.npz"
    handle.save_overlay(str(overlay))
    reloaded = atomicng.MemoryPatch.load_pretrained(str(overlay), spec)
    assert reloaded.checksum() == handle.memory.checksum()

    handle.memory.clear()
    assert torch.equal(base, logits_now()), "cleaning Δ did not return the attached base (bitbitbit))"
    handle.detach()
