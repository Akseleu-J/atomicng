"""Тесты моста к vLLM: merged-таблица, плагин entry-point, export-артефакты.

vllm как зависимость НЕ нужен: плагин вызывается движком, а не наоборот.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atomicng
from atomicng import TinyModel
from atomicng.vllm_bridge import (
    OVERLAY_ENV,
    config_for_vllm,
    export_for_vllm,
    merged_table,
    merged_table_from_overlay,
    vllm_general_plugins,
)


def _model_with_overlay(tmp_path):
    model = TinyModel(spec=atomicng.TableSpec(name="vllm", n_rows=1024, row_width=16))
    writer = atomicng.HashWriter(model.spec)
    atomicng.write(model, "vllm bridge overlay document about kangaroos", writer=writer)
    overlay = tmp_path / "mem.atomicng.npz"
    model.memory.save_pretrained(str(overlay))
    return model, str(overlay)


def test_merged_table_equals_base_plus_delta(tmp_path):
    model, overlay = _model_with_overlay(tmp_path)
    merged, patch = merged_table_from_overlay(model.t0, overlay, model.spec)
    assert merged.shape == model.t0.shape
    delta_rows = set(patch.row_ids())
    assert any(not np.array_equal(merged[r], model.t0[r]) for r in delta_rows)
    clean = [r for r in range(0, 50) if r not in delta_rows]
    assert all(np.array_equal(merged[r], model.t0[r]) for r in clean)


def test_merged_table_shape_mismatch_rejected(tmp_path):
    model, overlay = _model_with_overlay(tmp_path)
    wrong = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        merged_table(wrong, model.memory)


def test_plugin_env_provider_and_absence(monkeypatch, tmp_path):
    model, overlay = _model_with_overlay(tmp_path)
    monkeypatch.delenv(OVERLAY_ENV, raising=False)
    assert vllm_general_plugins() == {}  # без оверлея плагин безвреден

    monkeypatch.setenv(OVERLAY_ENV, overlay)
    plugins = vllm_general_plugins()
    assert "atomicng_ngram_overlay" in plugins
    provider = plugins["atomicng_ngram_overlay"]
    merged, patch = provider(model.t0, model.spec)
    assert merged.shape == model.t0.shape and len(patch) > 0

    monkeypatch.setenv(OVERLAY_ENV, str(tmp_path / "missing.npz"))
    with pytest.raises(FileNotFoundError):
        vllm_general_plugins()


def test_export_for_vllm_artifacts(tmp_path):
    model, overlay = _model_with_overlay(tmp_path)
    out = export_for_vllm(model.t0, model.memory, str(tmp_path / "vllm"), model_id="test-model")
    assert (out / "ngram_table_merged.npz").exists()
    cfg = json.loads((out / "vllm_config.json").read_text(encoding="utf-8"))
    assert cfg["model"] == "test-model" and cfg["n_delta_rows"] == len(model.memory)
    assert len(cfg["sha256_table"]) == 64
    loaded = np.load(out / "ngram_table_merged.npz")["table"]
    assert loaded.shape == model.t0.shape


def test_config_for_vllm_fields(tmp_path):
    model, overlay = _model_with_overlay(tmp_path)
    cfg = config_for_vllm(model.memory, "m", "x.npz", overlay_path=overlay)
    assert cfg["overlay"] == overlay and cfg["bridge_version"] == 1
    assert config_for_vllm(model.memory, "m", "x.npz")["overlay"] is None
    assert cfg["spec"]["name"] == "vllm"


def test_entry_point_registered():
    """Entry-point vllm.general_plugins объявлен в метаданных пакета."""
    from importlib import metadata

    eps = [ep for ep in metadata.entry_points(group="vllm.general_plugins")
           if ep.name == "atomicng_ngram_overlay" and "atomicng" in str(ep.value)]
    assert eps, "entry-point vllm.general_plugins не зарегистрирован"
    assert eps[0].load() is vllm_general_plugins
