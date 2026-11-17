"""Run the released AtomicNG Writer with SmolLM2-135M on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from atomicng import TableSpec
from atomicng.backends.hf import attach_sidecar
from atomicng.writers.io import load_writer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
WRITER_URL = "https://github.com/Akseleu-J/atomicng/releases/download/v0.5.0a2/writer_night.safetensors"
WRITER_SHA256 = "184952f3f6c5705786feecd64121fe6ed907117f3b55f0f2e09e1fba29cf3fc8"
EVIDENCE = (
    "The field station stores its volcanic soil samples in cabinet seven. "
    "Cabinet seven contains the field station's volcanic soil samples."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Use a local copy of the released checkpoint.")
    parser.add_argument("--local-files-only", action="store_true", help="Use a cached base model.")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = Path(directory) / "writer_night.safetensors"
            with urllib.request.urlopen(WRITER_URL, timeout=30) as response:
                checkpoint.write_bytes(response.read(1_000_000))
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != WRITER_SHA256:
            raise SystemExit("Checkpoint SHA-256 does not match the published Writer.")
        writer, config = load_writer(str(checkpoint))

    torch.set_num_threads(2)
    options = {"revision": MODEL_REVISION, "local_files_only": args.local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **options).float().eval()
    spec = TableSpec(name=config["spec_name"], n_rows=4096, row_width=config["row_width"])

    # The original run did not save its projection. Initialize this demo's projection
    # reproducibly without changing the caller's CPU RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        handle = attach_sidecar(model, spec=spec, inject_layer=2, seed=17)

    ids = tokenizer.encode(EVIDENCE, add_special_tokens=False)
    input_ids = torch.tensor([ids])
    with torch.inference_mode():
        try:
            with handle.write_session(ids):
                before = model(input_ids=input_ids).logits.contiguous().numpy().tobytes()
            nll_before = handle.nll([ids])

            handle.write(ids, writer, memory_aware=config["memory_aware"])
            rows = len(handle.memory)
            with handle.write_session(ids):
                written = model(input_ids=input_ids).logits.contiguous().numpy().tobytes()
            nll_written = handle.nll([ids])

            handle.memory.clear()
            with handle.write_session(ids):
                cleared = model(input_ids=input_ids).logits.contiguous().numpy().tobytes()
        finally:
            handle.detach()

    result = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "writer_sha256": WRITER_SHA256,
        "writer_training_steps": config["steps"],
        "rows_written": rows,
        "output_changed": written != before,
        "rollback_bit_exact": cleared == before,
        "nll_before": nll_before,
        "nll_written": nll_written,
    }
    print(json.dumps(result, indent=2))
    if not (rows > 0 and result["output_changed"] and result["rollback_bit_exact"]):
        raise SystemExit("Writer smoke check failed.")


if __name__ == "__main__":
    main()
