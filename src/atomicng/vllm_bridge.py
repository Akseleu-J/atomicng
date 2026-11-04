"""Мост к vLLM: нативная поддержка оверлеев через официальный механизм плагинов.

Три пути интеграции (docs/vllm.md):
1. **Merged table** — `atomicng export-vllm` собирает полную таблицу T⁰+Δ (npz/safetensors
   + config.json), vLLM-раннер подмешивает её при загрузке модели;
2. **general plugin** — пакет регистрирует entry-point `vllm.general_plugins`; при старте
   `vllm serve` плагин читает `VLLM_ATOMICNG_OVERLAY` и отдаёт провайдер merged-таблицы;
3. **upstream Engram-путь** — PR в vLLM для моделей с заводской таблицей (Track B, v1.0).

Модуль НЕ импортирует vllm (плагин вызывается движком, а не наоборот) — зависимость
односторонняя, Windows-разработка и тесты возможны без vLLM.
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
    """Загружает .dng-оверлей для движка (строгая валидация версии/спеки)."""
    return MemoryPatch.load_pretrained(path, spec)


def merged_table(base_table: np.ndarray, patch: MemoryPatch) -> np.ndarray:
    """Полная таблица T⁰+Δ (материализованный патч) — то, что подмешивает движок."""
    full = np.asarray(base_table, dtype=np.float32).copy()
    if full.shape != (patch.spec.n_rows, patch.spec.row_width):
        raise ValueError(
            f"базовая таблица {full.shape} не совпадает со спекой "
            f"({patch.spec.n_rows}, {patch.spec.row_width})"
        )
    patch.materialize()
    for r in patch.row_ids():
        full[r] += patch._rows[r]
    return full


def merged_table_from_overlay(
    base_table: np.ndarray, overlay_path: str, spec: TableSpec | None = None
) -> tuple[np.ndarray, MemoryPatch]:
    """ Convenience для плагина: оверлей с диска → merged таблица."""
    patch = load_overlay(overlay_path, spec)
    return merged_table(base_table, patch), patch


def config_for_vllm(
    patch: MemoryPatch, model_id: str, merged_path: str, overlay_path: str | None = None
) -> dict:
    """Манифест для vLLM-раннера: что подмешивать и откуда."""
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
    """Собирает артефакты для vLLM: merged таблицу (.npz) + config.json. Возвращает каталог."""
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
    """Entry-point `vllm.general_plugins` (vLLM ≥ 0.6): движок вызывает при старте.

    Если задан VLLM_ATOMICNG_OVERLAY — возвращает провайдер merged-таблицы; иначе пустой dict
    (плагин безвреден при отсутствии оверлея).
    """
    overlay = os.environ.get(OVERLAY_ENV)
    if not overlay:
        return {}
    if not Path(overlay).exists():
        raise FileNotFoundError(
            f"{OVERLAY_ENV} указывает на несуществующий оверлей: {overlay!r}"
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
