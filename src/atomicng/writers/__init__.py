"""Writer'ы (плагин-слой, T3): превращают признаки событий в цели и гейты.

Требуют torch; ядро atomicng от torch не зависит. Импортируйте осознанно:
    from atomicng.writers import WriterMLP
"""
from __future__ import annotations

from ..exceptions import BackendError


def require_torch() -> None:
    try:
        import torch  # noqa: F401, PLC0415
    except ImportError as exc:
        raise BackendError("слой writer'ов требует torch: pip install 'atomicng[torch]'") from exc


def get_writer(name: str, **kwargs):
    """Фабрика writer'ов: 'plain' | 'delta' (полная — из статьи; 'tiny-hash' — ядро без torch)."""
    require_torch()
    if name in ("plain", "delta"):
        from .mlp import WriterMLP  # noqa: PLC0415

        return WriterMLP(memory_aware=name == "delta", **kwargs)
    raise ValueError(f"неизвестный writer {name!r} (доступны: plain, delta)")


__all__ = ["WriterMLP", "get_writer", "require_torch"]
