"""Артефакты writer'а: safetensors (data-only, без pickle — S1), конфиг в метаданных.

Формат: safetensors-файл с state_dict writer'а; метаданные (архитектура, гиперпараметры,
версия формата) — в metadata-блоке safetensors.
"""
from __future__ import annotations

import hashlib
import json

from ..exceptions import BackendError, OverlayError

WRITER_FORMAT = "atomicng-writer"
WRITER_VERSION = 1


def _require():
    try:
        from safetensors import safe_open  # noqa: PLC0415
        from safetensors.torch import save_file  # noqa: PLC0415
    except ImportError as exc:
        raise BackendError("writers.io требует torch и safetensors: pip install 'atomicng[torch]'") from exc
    return safe_open, save_file


def writer_config(writer) -> dict:
    """Конфиг архитектуры WriterMLP для метаданных артефакта."""
    head_embed_dim = int(writer.head_embed.embedding_dim)
    feature_dim = int(writer.v_head[0].in_features)
    return {
        "arch": "WriterMLP",
        "memory_aware": bool(writer.memory_aware),
        "target_scale": float(writer.target_scale),
        "bound_targets": bool(getattr(writer, "bound_targets", False)),
        "hidden": int(writer.v_head[0].out_features),
        "row_width": int(writer.row_width),
        "hidden_size": (feature_dim - head_embed_dim) // 3,
        "head_embed_dim": head_embed_dim,
    }


def save_writer(writer, path: str, extra: dict | None = None) -> None:
    """Сохраняет writer (state_dict + конфиг в метаданных)."""
    _safe_open, save_file = _require()
    config = writer_config(writer)
    config.update(extra or {})
    config.update({"writer_format": WRITER_FORMAT, "writer_version": WRITER_VERSION})
    state = {k: v.detach().contiguous().cpu() for k, v in writer.state_dict().items()}
    save_file(state, path, metadata={k: json.dumps(v) for k, v in config.items()})


def load_writer(path: str, writer=None):
    """Загружает writer; без переданного модуля — строит по конфигу из метаданных.

    Возвращает (writer, config). Неизвестная версия формата — отказ.
    """
    safe_open, _save_file = _require()
    with safe_open(path, framework="pt") as f:
        raw_metadata = f.metadata() or {}
        state = {k: f.get_tensor(k) for k in f.keys()}
    try:
        config = {k: json.loads(v) for k, v in raw_metadata.items()}
    except (TypeError, json.JSONDecodeError) as exc:
        raise OverlayError(f"writer {path!r}: нечитаемые метаданные ({exc})") from exc
    if config.get("writer_format") != WRITER_FORMAT or int(config.get("writer_version", -1)) != WRITER_VERSION:
        raise OverlayError(
            f"writer {path!r}: формат {config.get('writer_format')!r} "
            f"v{config.get('writer_version')} не поддерживается"
        )
    if writer is None:
        if config.get("arch") != "WriterMLP":
            raise OverlayError(f"writer {path!r}: неизвестная архитектура {config.get('arch')!r}")
        from .mlp import WriterMLP  # noqa: PLC0415

        writer = WriterMLP(
            hidden_size=int(config["hidden_size"]),
            row_width=int(config["row_width"]),
            hidden=int(config["hidden"]),
            target_scale=float(config["target_scale"]),
            head_embed_dim=int(config.get("head_embed_dim", 16)),
            memory_aware=bool(config["memory_aware"]),
            bound_targets=bool(config.get("bound_targets", False)),
        )
    writer.load_state_dict(state)
    writer.eval()
    return writer, config


def writer_fingerprint(writer) -> str:
    """sha256 state_dict (для provenance: каким writer'ом писали)."""
    h = hashlib.sha256()
    state = writer.state_dict()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()
