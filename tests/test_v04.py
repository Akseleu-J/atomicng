"""v0.4.0a1: plain-режим кодирования (T⁰-only), HTML-монитор, PLE-экспорт."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atomicng
from atomicng import TinyModel

# ------------------------------------------------- plain-кодирование (статья)


def test_plain_mode_encodes_on_t0_only():
    """plain-вариант статьи: кодирование не видит текущую Δ (нулевые Δ_j и c_j)."""
    pytest.importorskip("torch")
    from atomicng.backends.hf import attach_sidecar, stub_causal_lm
    from atomicng.writers.mlp import WriterMLP

    model = stub_causal_lm(hidden_size=32, vocab_size=64, n_layers=4)
    spec = atomicng.TableSpec(name="stub-plain", n_rows=512, row_width=32)
    handle = attach_sidecar(model, spec=spec, inject_layer=1)
    writer = WriterMLP(hidden_size=32, row_width=32, hidden=64, target_scale=0.3)

    seen_deltas: list = []
    seen_counts: list = []
    original_forward = writer.forward

    def spy_forward(hidden, row_ids, delta_rows, counts):
        seen_deltas.append(delta_rows.detach().clone())
        seen_counts.append(counts.detach().clone())
        return original_forward(hidden, row_ids, delta_rows, counts)

    writer.forward = spy_forward  # instance-атрибут перекрывает nn.Module.forward

    ids = [5, 17, 3, 29, 11, 41, 5, 17]
    handle.write(ids, writer, memory_aware=True)   # первая запись: память пуста
    assert seen_deltas[0].abs().sum() == 0
    handle.write(ids, writer, memory_aware=True)   # вторая: delta-режим ВИДИТ Δ
    assert seen_deltas[1].abs().sum() > 0
    handle.write(ids, writer, memory_aware=False)  # plain-режим: Δ и c_j нулевые в кодировании
    assert seen_deltas[2].abs().sum() == 0
    assert seen_counts[2].abs().sum() == 0
    assert len(handle.memory) > 0, "plain-кодирование не отменяет саму запись"
    handle.detach()


# ------------------------------------------------------------- HTML-монитор


def test_monitor_report_contains_sections(tmp_path):
    model = TinyModel()
    writer = atomicng.HashWriter(model.spec)
    reg = atomicng.MetricsRegistry()
    for i in range(3):
        atomicng.write(model, f"monitor test document {i} with facts", writer=writer, metrics=reg)
    html_text = atomicng.build_report(model.memory, metrics_records=reg.records)
    for marker in ("AtomicNG Monitor", "Состояние памяти", "Гистограмма норм", "Метрики записи", "svg"):
        assert marker in html_text, f"нет секции {marker!r}"
    assert "<script" not in html_text, "отчёт обязан быть без внешних скриптов"

    out = tmp_path / "report.html"
    atomicng.write_report(str(out), patch=model.memory, metrics_records=reg.records,
                          verify_message="откат побитовый", verify_passed=True)
    assert out.exists() and out.stat().st_size > 2000


def test_monitor_metrics_jsonl_roundtrip(tmp_path):
    model = TinyModel()
    reg = atomicng.MetricsRegistry()
    atomicng.write(model, "jsonl metrics doc", metrics=reg)
    path = tmp_path / "m.jsonl"
    reg.export_jsonl(str(path))
    records = atomicng.load_metrics_jsonl(str(path))
    assert len(records) == 1 and "mean_gate" in records[0]
    html_text = atomicng.build_report(model.memory, metrics_records=records)
    assert "записей" in html_text


# ------------------------------------------------------------- PLE-экспорт


def test_ple_export_roundtrip_and_delta_effect(tmp_path):
    model = TinyModel(spec=atomicng.TableSpec(name="ple", n_rows=1024, row_width=16))
    writer = atomicng.HashWriter(model.spec)
    atomicng.write(model, "ple export document about kangaroos", writer=writer)
    out = tmp_path / "table.ple"
    integrity = atomicng.export_ple(model.t0, model.memory, str(out), source="test")
    assert integrity and out.exists()

    spec, table, header = atomicng.load_ple(str(out))
    assert spec == model.spec and table.shape == (1024, 16)
    assert header["source"] == "test" and header["n_delta_rows"] == len(model.memory)
    # строки с Δ отличаются от базы; строки без Δ совпадают побитово
    delta_rows = set(model.memory.row_ids())
    assert any(not np.array_equal(table[r], model.t0[r]) for r in delta_rows)
    clean = [r for r in (0, 5, 100) if r not in delta_rows]
    assert all(np.array_equal(table[r], model.t0[r]) for r in clean)


def test_ple_rejects_corruption_and_wrong_magic(tmp_path):
    model = TinyModel(spec=atomicng.TableSpec(name="ple2", n_rows=512, row_width=8))
    out = tmp_path / "t.ple"
    atomicng.export_ple(model.t0, model.memory, str(out))
    data = bytearray(out.read_bytes())
    data[-1] ^= 0xFF  # порча тела
    out.write_bytes(bytes(data))
    with pytest.raises(atomicng.OverlayError):
        atomicng.load_ple(str(out))

    bad = tmp_path / "bad.ple"
    bad.write_bytes(b"NOTPLE" + b"\x00" * 64)
    with pytest.raises(atomicng.OverlayError):
        atomicng.load_ple(str(bad))


def test_ple_base_shape_mismatch_rejected(tmp_path):
    model = TinyModel(spec=atomicng.TableSpec(name="ple3", n_rows=512, row_width=8))
    wrong = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(atomicng.OverlayError):
        atomicng.export_ple(wrong, model.memory, str(tmp_path / "x.ple"))
