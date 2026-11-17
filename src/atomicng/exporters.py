"""Export the merged table T⁰+Δ in binary PLE format.

The overlay stores only Δ, so export requires the source table T⁰.
The file contains the DNGPLE01 magic bytes, a little-endian uint32 JSON
header length, the JSON header, and little-endian float32 rows in row-major order.
The header records the specification, provenance, export version, and SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from .exceptions import OverlayError
from .memory import MemoryPatch
from .specs import TableSpec

PLE_MAGIC = b"DNGPLE01"
PLE_EXPORT_VERSION = 1


def _materialize_full_table(base_table: np.ndarray, patch: MemoryPatch) -> np.ndarray:
    """T⁰ + Δ (materialized) - complete table for the engine."""
    full = np.asarray(base_table, dtype=np.float32).copy()
    if full.shape != (patch.spec.n_rows, patch.spec.row_width):
        raise OverlayError(
            f"base-table {full.shape} not matching the stew "
            f"({patch.spec.n_rows}, {patch.spec.row_width})"
        )
    patch.materialize()
    for r in patch.row_ids():
        full[r] += patch._rows[r]
    return full


def export_ple(base_table: np.ndarray, patch: MemoryPatch, path: str, *, source: str = "sidecar") -> str:
    """Write a PLE file with the merged table T⁰+Δ and return its SHA-256 digest."""
    full = _materialize_full_table(base_table, patch)
    header = {
        "export_version": PLE_EXPORT_VERSION,
        "spec": json.loads(patch.spec.to_json()),
        "source": source,
        "n_rows": patch.spec.n_rows,
        "row_width": patch.spec.row_width,
        "dtype": "float32-le",
        "created_by": patch.created_by,
        "sources": patch.sources,
        "n_delta_rows": len(patch),
    }
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    body = full.astype("<f4").tobytes(order="C")
    integrity = hashlib.sha256(PLE_MAGIC + header_bytes + body).hexdigest()
    header["integrity_sha256"] = integrity
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    with open(path, "wb") as f:
        f.write(PLE_MAGIC)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(body)
    return integrity


def load_ple(path: str) -> tuple[TableSpec, np.ndarray, dict]:
    """Reads the PLE file → (speak, full table, header).."""
    data = Path(path).read_bytes()
    if data[: len(PLE_MAGIC)] != PLE_MAGIC:
        raise OverlayError(f"{path!r}: invalid PLE file magic")
    (header_len,) = struct.unpack_from("<I", data, len(PLE_MAGIC))
    off = len(PLE_MAGIC) + 4
    header = json.loads(data[off : off + header_len].decode("utf-8"))
    body = data[off + header_len :]
    expected = header.get("integrity_sha256")
    if expected is not None:
        no_integrity = {k: v for k, v in header.items() if k != "integrity_sha256"}
        header_bytes = json.dumps(no_integrity, ensure_ascii=False).encode("utf-8")
        actual = hashlib.sha256(PLE_MAGIC + header_bytes + body).hexdigest()
        if actual != expected:
            raise OverlayError(f"{path!r}: PLE integrity check failed")
    spec = TableSpec.from_json(json.dumps(header["spec"]))
    table = np.frombuffer(body, dtype="<f4").reshape(header["n_rows"], header["row_width"]).copy()
    return spec, table, header
