"""Writer’s artifacts: safetensors round-trip, strict validation, fingerprint."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomicng.writers.io import load_writer, save_writer, writer_fingerprint
from atomicng.writers.mlp import WriterMLP


def test_save_load_roundtrip(tmp_path):
    writer = WriterMLP(hidden_size=32, row_width=16, hidden=64, memory_aware=False)
    path = str(tmp_path / "writer.safetensors")
    save_writer(writer, path, extra={"spec_name": "sidecar-small"})
    loaded, config = load_writer(path)
    assert config["arch"] == "WriterMLP" and config["hidden_size"] == 32
    assert config["row_width"] == 16 and config["spec_name"] == "sidecar-small"
    for k, v in writer.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])
    assert writer_fingerprint(writer) == writer_fingerprint(loaded)


def test_rejects_unknown_version(tmp_path):
    from safetensors.torch import save_file

    writer = WriterMLP(hidden_size=32, row_width=16, hidden=64)
    path = str(tmp_path / "bad.safetensors")
    state = {k: v.contiguous() for k, v in writer.state_dict().items()}
    save_file(state, path, metadata={"writer_format": '"atomicng-writer"', "writer_version": "999"})
    with pytest.raises(atomicng_overlay_error()):
        load_writer(path)


def atomicng_overlay_error():
    import atomicng

    return atomicng.OverlayError


def test_fingerprint_distinguishes_writers():
    a = WriterMLP(hidden_size=32, row_width=16, hidden=64)
    b = WriterMLP(hidden_size=32, row_width=16, hidden=64)
    assert writer_fingerprint(a) != writer_fingerprint(b)
