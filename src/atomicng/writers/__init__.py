"""Writer components that turn event features into row targets and gates.

These components require the optional PyTorch layer.
"""
from __future__ import annotations

from ..exceptions import BackendError


def require_torch() -> None:
    try:
        import torch  # noqa: F401, PLC0415
    except ImportError as exc:
        raise BackendError("The writer layer requires torch: pip install 'atomicng[torch]'") from exc


def get_writer(name: str, **kwargs):
    """Writers factory: ’plain' | 'delta’ (complete - from the article; ’tiny-hash’ - kernel without torch))."""
    require_torch()
    if name in ("plain", "delta"):
        from .mlp import WriterMLP  # noqa: PLC0415

        return WriterMLP(memory_aware=name == "delta", **kwargs)
    raise ValueError(f"Unknown writer {name!r}. Available writers: plain and delta.")


__all__ = ["WriterMLP", "get_writer", "require_torch"]
