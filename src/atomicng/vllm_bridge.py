"""Export a merged n-gram table and expose it through the vLLM plugin entry point.

The plugin reads VLLM_ATOMICNG_OVERLAY and provides the merged-table loader.
See docs/vllm.md for setup and the upstream Engram proposal. This module can
be imported and tested without vLLM.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .memory import MemoryPatch
from .specs import TableSpec

OVERLAY_ENV = "VLLM_ATOMICNG_OVERLAY"
BRIDGE_VERSION = 1


def load_overlay(path: str, spec: TableSpec | None = None) -> MemoryPatch:
    """Load a .dng overlay for the engine and validate its version and table specification."""
    return MemoryPatch.load_pretrained(path, spec)


def merged_table(base_table: np.ndarray, patch: MemoryPatch) -> np.ndarray:
    """Return the materialized T⁰+Δ table for the engine."""
    full = np.asarray(base_table, dtype=np.float32).copy()
    if full.shape != (patch.spec.n_rows, patch.spec.row_width):
        raise ValueError(
            f"base-table {full.shape} not matching the stew "
            f"({patch.spec.n_rows}, {patch.spec.row_width})"
        )
    patch.materialize()
    for r in patch.row_ids():
        full[r] += patch._rows[r]
    return full


def merged_table_from_overlay(
    base_table: np.ndarray, overlay_path: str, spec: TableSpec | None = None
) -> tuple[np.ndarray, MemoryPatch]:
    """ Convenience for plugin: overlays from disk → merged table."""
    patch = load_overlay(overlay_path, spec)
    return merged_table(base_table, patch), patch


def config_for_vllm(
    patch: MemoryPatch, model_id: str, merged_path: str, overlay_path: str | None = None
) -> dict:
    """Manifesto for vLLM-runner: what to mix and where to get it."""
    return {
        "bridge_version": BRIDGE_VERSION,
        "model": model_id,
        "overlay": overlay_path,
        "merged_table": str(merged_path),
        "spec": json.loads(patch.spec.to_json()),
        "n_delta_rows": len(patch),
        "total_writes": patch.total_writes,
        "decay": patch.decay,
        "sources": patch.sources,
    }


def export_for_vllm(
    base_table: np.ndarray, patch: MemoryPatch, out_dir: str, *,
    model_id: str, overlay_path: str | None = None,
) -> Path:
    """Write the merged NPZ table and vLLM configuration and return the output directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged = merged_table(base_table, patch)
    merged_path = out / "ngram_table_merged.npz"
    np.savez(merged_path, table=merged)
    cfg = config_for_vllm(patch, model_id, str(merged_path), overlay_path)
    cfg["sha256_table"] = __import__("hashlib").sha256(merged.tobytes()).hexdigest()
    (out / "vllm_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def vllm_general_plugins() -> dict:
    """Entry-point `vllm.general_plugins` (vLLM ≥ 0.6: engine calls at start.

    If given the VLLM_ATOMICNG_OVERLAY — returns the merged table provider; otherwise, the empty dict
    (Plugin is harmless in the absence of overlay).
    """
    overlay = os.environ.get(OVERLAY_ENV)
    if not overlay:
        return {}
    if not Path(overlay).exists():
        raise FileNotFoundError(
            f"{OVERLAY_ENV} pointing to a non-existent overlay: {overlay!r}"
        )

    def atomicng_merged_table_provider(base_table: np.ndarray, spec: TableSpec | None = None):
        return merged_table_from_overlay(base_table, overlay, spec)

    return {"atomicng_ngram_overlay": atomicng_merged_table_provider}


__all__ = [
    "OVERLAY_ENV",
    "config_for_vllm",
    "export_for_vllm",
    "load_overlay",
    "merged_table",
    "merged_table_from_overlay",
    "vllm_general_plugins",
]
