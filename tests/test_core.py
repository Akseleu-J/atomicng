"""Property-тесты ядра atomicng v0.1 (запуск: pytest или python tests/run_all.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atomicng
from atomicng import SIDECAR_SMALL, D1Rule, Event, MemoryPatch, TinyModel


def _rng_events(rng, n_rows: int, width: int, n: int, gate_range=(0.0, 1.0)) -> list[Event]:
    return [
        Event(
            row_id=int(rng.integers(0, n_rows)),
            target=(0.05 * rng.standard_normal(width)).astype(np.float32),
            gate=float(rng.uniform(*gate_range)),
        )
        for _ in range(n)
    ]


# ---------------------------------------------------------------- свойства D1


def test_single_event_is_convex_combination():
    """Одинокая запись с гейтом β: Δ⁺ = (1−β)·Δ + β·v (следствие D1)."""
    patch = MemoryPatch(SIDECAR_SMALL)
    v = np.full(SIDECAR_SMALL.row_width, 0.2, dtype=np.float32)
    patch.bump_clock()
    patch._write_row(7, np.full(SIDECAR_SMALL.row_width, 0.6, dtype=np.float32))
    D1Rule().apply(patch, [Event(7, v, 0.3)])
    expected = 0.7 * 0.6 + 0.3 * 0.2
    assert np.allclose(patch.get(7), expected, atol=1e-6)


def test_block_averaging_matches_paper_form():
    """Δ⁺ = (1−w)·Δ + mean(β_a·v_a), w = mean(β_a) — блочно, не последовательно."""
    patch = MemoryPatch(SIDECAR_SMALL)
    rng = np.random.default_rng(3)
    v1, v2 = (0.1 * rng.standard_normal(SIDECAR_SMALL.row_width).astype(np.float32) for _ in range(2))
    b1, b2 = 0.8, 0.4
    before = np.full(SIDECAR_SMALL.row_width, 0.5, dtype=np.float32)
    patch.bump_clock()
    patch._write_row(11, before)
    D1Rule().apply(patch, [Event(11, v1, b1), Event(11, v2, b2)])
    a = (b1 * v1 + b2 * v2) / 2
    w = (b1 + b2) / 2
    assert np.allclose(patch.get(11), (1 - w) * before + a, atol=1e-6)


def test_bounded_norm_property():
    """Prop. 1: ‖Δ_j⁺‖ ≤ max(‖Δ_j‖, max_a ‖v_a‖) на длинной случайной цепочке блоков."""
    rng = np.random.default_rng(17)
    patch = MemoryPatch(SIDECAR_SMALL)
    row = 42
    for _ in range(50):
        events = [Event(row, t, g) for t, g in (
            ((0.05 * rng.standard_normal(SIDECAR_SMALL.row_width)).astype(np.float32),
             float(rng.uniform(0, 1))) for _ in range(rng.integers(1, 5))
        )]
        max_target_norm = max(float(np.linalg.norm(e.target)) for e in events)
        before_norm = float(np.linalg.norm(patch.get(row)))
        D1Rule().apply(patch, events)
        after_norm = float(np.linalg.norm(patch.get(row)))
        assert after_norm <= max(before_norm, max_target_norm) + 1e-6


def test_block_order_invariance():
    """Порядок событий внутри блока не влияет на результат (среднее коммутативно)."""
    rng = np.random.default_rng(5)
    events = _rng_events(rng, SIDECAR_SMALL.n_rows, SIDECAR_SMALL.row_width, 30)
    p1, p2 = MemoryPatch(SIDECAR_SMALL), MemoryPatch(SIDECAR_SMALL)
    D1Rule().apply(p1, events)
    D1Rule().apply(p2, list(reversed(events)))
    assert p1.checksum() == p2.checksum()


def test_gate_validation():
    patch = MemoryPatch(SIDECAR_SMALL)
    bad = Event(1, np.zeros(SIDECAR_SMALL.row_width, np.float32), 1.5)
    try:
        D1Rule().apply(patch, [bad])
    except ValueError:
        return
    raise AssertionError("гейт 1.5 должен отвергаться")


def test_hot_row_guard_warns():
    patch = MemoryPatch(SIDECAR_SMALL)
    target = np.full(SIDECAR_SMALL.row_width, 0.01, np.float32)
    report = D1Rule().apply(patch, [Event(3, target, 1.0)], atomicng.WritePolicies(hot_row_norm_cap=0.001))
    assert report.rows[0].hot and report.warnings


# ---------------------------------------------------------- неразрушаемость


def _model_with_memory() -> TinyModel:
    model = TinyModel()
    return model


def test_write_locality_query_effect_and_bit_exact_deletion():
    """Локальность: нейтральные пробы не меняются от записи (побитово).
    Эффект: запрос, содержащий записанные биграммы, меняет выход. verify проходит."""
    model = _model_with_memory()
    writer = atomicng.HashWriter(model.spec)
    evidence = "A fact about volcanic soil chemistry."
    neutral = model.probe(n=16)
    query = writer.tokenize(evidence)
    q_base = model.logits(query)

    atomicng.write(model, evidence, writer=writer)

    assert np.array_equal(neutral, model.probe(n=16)), "запись повлияла на нейтральные пробы (нет локальности)"
    q_after_write = model.logits(query)
    assert not np.array_equal(q_base, q_after_write), "запись не повлияла на запрос с записанными биграммами"

    report = atomicng.verify_deletion(model, n_probes=16)
    assert report.passed and report.max_abs_diff == 0.0
    assert np.array_equal(model.logits(query), q_after_write), "restore после verify исказил эффект памяти"


def test_revert_is_exact():
    """Pre-image journal: revert возвращает состояние байт-в-байт (к состоянию после write 1)."""
    model = _model_with_memory()
    undo = atomicng.UndoJournal()
    atomicng.write(model, ["doc one", "doc two"], undo=undo)
    checksum_after_first = model.memory.checksum()
    atomicng.write(model, "doc three", undo=undo)
    assert len(undo) == 2 and len(model.memory) > 0
    atomicng.revert_last(model, undo)
    assert model.memory.checksum() == checksum_after_first
    assert atomicng.verify_deletion(model).passed


def test_overlay_roundtrip():
    """Сохранение/загрузка .atomicng без потерь (строгая валидация версии формата)."""
    import tempfile

    model = _model_with_memory()
    atomicng.write(model, "overlay roundtrip document")
    probe_saved = model.probe()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "overlay.atomicng.npz")
        model.memory.save_pretrained(path)
        reloaded = MemoryPatch.load_pretrained(path, model.spec)
        assert reloaded.checksum() == model.memory.checksum()
    model.memory.clear()
    assert np.array_equal(model.probe(), TinyModel().probe())
    model.memory.restore = None  # sanity: дальше работаем с загруженным
    patch = reloaded
    fresh = _model_with_memory()
    fresh.memory = patch
    assert np.array_equal(fresh.probe(), probe_saved)


def test_rejects_unknown_overlay_version():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.atomicng.npz"
        np.savez(path, row_ids=np.zeros(0, np.int64), deltas=np.zeros((0, 160), np.float32),
                 counts=np.zeros(0, np.int64), meta=np.asarray(json.dumps({"overlay_format": "atomicng-overlay", "overlay_version": 999})))
        try:
            MemoryPatch.load_pretrained(str(path))
        except atomicng.OverlayError:
            return
    raise AssertionError("неизвестная версия оверлея должна отвергаться")


def test_audit_log_records_writes(tmp_path=None):
    import tempfile

    model = _model_with_memory()
    with tempfile.TemporaryDirectory() as tmp:
        log = atomicng.AuditLog(str(Path(tmp) / "audit.jsonl"))
        atomicng.write(model, "audited write", audit=log)
        atomicng.write(model, ["another", "write"], audit=log)
        assert len(log) == 2
        fresh = atomicng.AuditLog(str(Path(tmp) / "audit.jsonl"))
        assert len(fresh) == 2 and fresh.entries[0]["kind"] == "write"


def test_lazy_decay():
    """Ленивый decay: значение строки затухает с тактами, materialize фиксирует."""
    patch = MemoryPatch(SIDECAR_SMALL, decay=0.5)
    patch.bump_clock()
    v = np.ones(SIDECAR_SMALL.row_width, dtype=np.float32)
    D1Rule().apply(patch, [Event(9, v, 1.0)])
    assert np.allclose(patch.get(9), 1.0, atol=1e-7)
    patch.bump_clock()
    assert np.allclose(patch.get(9), 0.5, atol=1e-7)
    patch.bump_clock()
    patch.bump_clock()
    assert np.allclose(patch.get(9), 0.125, atol=1e-7)
