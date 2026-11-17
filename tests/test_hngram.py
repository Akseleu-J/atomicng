"""H-Ngram: L2-субтаблица поверх ядра AtomicNG (GPU-совместимые кернелы, CPU-fallback).

Без torch эти тесты пропускаются (pytest.importorskip) — как test_torch.py.
Инварианты PoC:

- zero-init gate: attach L2 не меняет выходы ядра бит-в-бит;
- домен: запись продолжений в субтаблицу снижает NLL на доменных пробах;
- нейтральные пробы (строки вне тома) не меняются вообще — негативный контроль;
- роутер с гистерезисом не thrash'ит на смешанных входах;
- LRU-менеджер подменяет тома без искажения значений.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomicng.exceptions import OverlayError  # noqa: E402
from atomicng.hngram.fill import fill_subtable_from_sequences  # noqa: E402
from atomicng.hngram.hashing import addresses, blake_addresses, poly_addresses  # noqa: E402
from atomicng.hngram.manager import SubtableManager  # noqa: E402
from atomicng.hngram.router import TopicRouter  # noqa: E402
from atomicng.hngram.sidecar import L2Sidecar  # noqa: E402
from atomicng.hngram.store import L2Subtable, SubtableSpec  # noqa: E402

VOCAB = 64
D = 32
N_ROWS = 256
SPEC = SubtableSpec(name="test-l2", n_rows=N_ROWS, row_width=D, key_order=2, hash_scheme="blake2b")

FACT = [5, 9, 13, 21, 33, 41]
NOISE = [2, 7, 11, 19, 23, 29, 31, 37, 43, 47]


def _corpus(n: int = 8, seed: int = 7) -> list[list[int]]:
    """Доменный корпус: факт FACT окружён шумом (стиль test_torch)."""
    gen = torch.Generator().manual_seed(seed)
    seqs = []
    for _ in range(n):
        head = [int(NOISE[int(torch.randint(0, len(NOISE), (1,), generator=gen))]) for _ in range(6)]
        tail = [int(NOISE[int(torch.randint(0, len(NOISE), (1,), generator=gen))]) for _ in range(6)]
        seqs.append(head + FACT + tail)
    return seqs


def _model(seed: int = 17):
    from atomicng.writers.training import TorchTinyModel

    return TorchTinyModel(vocab_size=VOCAB, hidden_size=D, n_rows=N_ROWS, seed=seed)


def _fill() -> L2Subtable:
    return fill_subtable_from_sequences(_corpus(), SPEC, VOCAB)


def _manager(tmp_path, *subs: L2Subtable, n_slots: int = 1) -> SubtableManager:
    manager = SubtableManager(str(tmp_path / "store"), n_slots=n_slots)
    for sub in subs:
        manager.register(sub)
    return manager


def _neutral_seqs(subtable: L2Subtable, n: int = 6, seq_len: int = 14, seed: int = 99) -> list[list[int]]:
    """Пробы, чьи биграммы не бьют ни в одну строку тома (негативный контроль)."""
    gen = torch.Generator().manual_seed(seed)
    occupied = set(torch.nonzero(subtable.occupancy).flatten().tolist())
    seqs, tries = [], 0
    while len(seqs) < n and tries < 5000:
        tries += 1
        seq = torch.randint(0, VOCAB, (seq_len,), generator=gen).tolist()
        if occupied.isdisjoint(blake_addresses(seq, N_ROWS, 2)):
            seqs.append([int(t) for t in seq])
    if len(seqs) < n:
        raise RuntimeError("Too dense subtable for neutral probes.")
    return seqs


def _nll(sidecar: L2Sidecar, seqs: list[list[int]]) -> float:
    """Средний NLL (nats/token) проб через sidecar."""
    total, count = 0.0, 0
    for seq in seqs:
        ids = torch.tensor([seq], dtype=torch.long)
        _, logits = sidecar(ids)
        total += float(torch.nn.functional.cross_entropy(logits[0], ids[0, 1:], reduction="sum").detach())
        count += len(seq) - 1
    return total / max(count, 1)


# ------------------------------------------------------------------ hashing


def test_blake_addresses_compatible_with_core():
    """blake2b-путь == TinyModel.row_hash (биграммы) — совместимость с ядром."""
    from atomicng.backends.tiny import TinyModel

    seq = [3, 17, 5, 200, 64]
    for w in range(len(seq) - 1):
        assert blake_addresses(seq, N_ROWS, 2)[w] == TinyModel.row_hash(seq[w], seq[w + 1], N_ROWS)


def test_poly_addresses_deterministic_and_aligned():
    """poly64: детерминизм, форма (B, T-k+1), диапазон [0, n_rows)."""
    ids = torch.randint(0, VOCAB, (3, 12), dtype=torch.long)
    a1 = poly_addresses(ids, N_ROWS, 2)
    a2 = poly_addresses(ids, N_ROWS, 2)
    assert torch.equal(a1, a2)
    assert a1.shape == (3, 11)
    assert (a1 >= 0).all() and (a1 < N_ROWS).all()
    assert poly_addresses(ids, N_ROWS, 3).shape == (3, 10)
    with pytest.raises(ValueError):
        poly_addresses(torch.randint(0, VOCAB, (1, 1)), N_ROWS, 3)


def test_poly_addresses_coverage():
    """Случайные окна заполняют таблицу широко (нет вырожденного схлопывания)."""
    ids = torch.randint(0, VOCAB, (64, 64), dtype=torch.long)
    rows = poly_addresses(ids, 1024, 2).flatten()
    assert len(set(rows.tolist())) >= 600  # теория: 1024·(1−e^(−4)) ≈ 980; с запасом


def test_addresses_dispatcher_rejects_unknown_scheme():
    """Диспетчер схем: неизвестная схема → отказ."""
    ids = torch.randint(0, VOCAB, (2, 10), dtype=torch.long)
    assert addresses(ids, N_ROWS, 2, "poly64").shape == (2, 9)
    with pytest.raises(ValueError, match="scheme"):
        addresses(ids, N_ROWS, 2, "md5")


# ------------------------------------------------------------------ store


def test_subtable_from_patch_lookup_and_miss():
    """Записанные строки читаются точно; промах → нули (откат к чистому L1)."""
    sub = _fill()
    rows = blake_addresses(FACT, N_ROWS, 2)
    vals, hits = sub.lookup(torch.tensor([rows]))
    assert hits.all()
    assert vals.shape == (1, len(rows), D)
    # строки, гарантированно свободные в томе (том плотный лишь частично)
    free = torch.nonzero(~sub.occupancy).flatten()[:3].tolist()
    miss, miss_hits = sub.lookup(torch.tensor([free]))
    assert not miss_hits.any() and bool((miss == 0).all())


def test_subtable_roundtrip_and_integrity(tmp_path):
    """save→load: значения совпадают в пределах int8-квантования; tamper → отказ."""
    import numpy as np

    sub = _fill()
    path = str(tmp_path / "l2.npz")
    sub.save_pretrained(path)
    loaded = L2Subtable.load_pretrained(path)
    assert loaded.spec == SPEC
    tol = max(float(sub.values.abs().max()) / 127.0, 1e-6)
    assert float((loaded.values - sub.values).abs().max()) <= tol

    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    payload["values_i8"][0, 0] += 7
    tampered = str(tmp_path / "tampered.npz")
    np.savez(tampered, **payload)
    with pytest.raises(OverlayError, match="integrity"):
        L2Subtable.load_pretrained(tampered)


def test_subtable_rejects_foreign_spec():
    """Несовпадающая геометрия патча и тома → OverlayError (строгая сборка)."""
    from atomicng.memory import MemoryPatch
    from atomicng.specs import TableSpec

    patch = MemoryPatch(TableSpec(name="p", n_rows=64, row_width=D))
    patch._write_row(3, torch.zeros(D).numpy())
    bad = SubtableSpec(name="bad", n_rows=N_ROWS, row_width=D + 8)
    with pytest.raises(OverlayError):
        L2Subtable.from_patch(patch, bad)


def test_hit_rate_probe_covers_filled_rows():
    """Пробы из корпуса записи бьют в том со 100% покрытием; случайные — на уровне плотности."""
    sub = _fill()
    # FACT-окна встречаются внутри каждого корпусного seq → все заняты
    fact_rows = blake_addresses(FACT, N_ROWS, 2)
    assert sub.hit_rate(torch.tensor(fact_rows)) == 1.0
    # окна того же корпуса (последнее окно seq не пишется — нет продолжения)
    same_corpus = [r for seq in _corpus(4) for r in blake_addresses(seq, N_ROWS, 2)[:-1]]
    assert sub.hit_rate(torch.tensor(same_corpus)) == 1.0
    rand_rows = torch.randint(0, N_ROWS, (256,), generator=torch.Generator().manual_seed(5))
    fill_ratio = sub.stats()["fill_ratio"]
    assert sub.hit_rate(rand_rows) < fill_ratio + 0.05  # случайная проба ≈ плотности тома


# ------------------------------------------------------------------ router


def test_router_hysteresis_no_thrash():
    """Гистерезис: слабые признаки не переключают тему; сильные — не thrash'ат."""
    gen = torch.Generator().manual_seed(3)
    router = TopicRouter(feature_dim=D, domains=["code", "med"], margin=0.5, cooldown=8, seed=17)
    # строки весов головы → детерминированно разные top-1 темы; margin масштабируется
    a = router.head.weight[0].detach()
    b = router.head.weight[1].detach()
    # фаза 1: слабые признаки (margin ≪ 0.5) → аргмакс «просит» смены, но гистерезис молчит
    for i in range(20):
        jitter = torch.randn(D, generator=gen) * 0.01
        router.route((a if i % 2 == 0 else b) * 0.2 + jitter, last_k=1)
    assert router.n_switches == 1  # только старт
    assert router.n_rejects >= 5  # b-шаги гасятся margin'ом и кулдауном
    assert router.thrashing() is False
    # фаза 2: сильные признаки (margin ≈ 1.3 ≫ 0.5) → смена происходит, кулдаун режет частоту
    for i in range(20):
        jitter = torch.randn(D, generator=gen) * 0.01
        router.route((a if i % 2 == 0 else b) * 100.0 + jitter, last_k=1)
    assert 2 <= router.n_switches <= 6  # без гистерезиса было бы ~39


def test_router_switches_on_clean_domain_change():
    """Чистая смена темы (после спада кулдауна) → переключение происходит."""
    router = TopicRouter(feature_dim=D, domains=["code", "med"], margin=1e-3, cooldown=1, seed=17)
    # признаки = строки весов головы ×10 → детерминированно разные top-1 темы
    x1 = router.head.weight[0].detach() * 10.0
    x2 = router.head.weight[1].detach() * 10.0
    first = router.route(x1)
    for _ in range(4):
        router.route(x1)
    last = router.route(x2)
    assert router.n_switches >= 2  # старт + смена темы
    assert first.name != last.name


# ------------------------------------------------------------------ manager


def _registered_manager(tmp_path, n_slots: int = 2) -> tuple[SubtableManager, dict[str, L2Subtable]]:
    manager = SubtableManager(str(tmp_path / "store"), n_slots=n_slots)
    subs = {}
    for i, name in enumerate(["code", "med", "law"]):
        spec = SubtableSpec(name=name, n_rows=N_ROWS, row_width=D, key_order=2, hash_scheme="blake2b")
        seqs = [[(t + 8 * i) % VOCAB for t in seq] for seq in _corpus(2, seed=7 + i)]
        subs[name] = fill_subtable_from_sequences(seqs, spec, VOCAB)
        manager.register(subs[name])
    return manager, subs


def test_manager_lru_eviction_and_reload(tmp_path):
    """LRU: третий том вытесняет первый; перезагрузка восстанавливает значения."""
    manager, subs = _registered_manager(tmp_path, n_slots=2)
    _, ev1, _ = manager.ensure("code")
    assert ev1 is None
    _, ev2, _ = manager.ensure("med")
    assert ev2 is None
    _, ev3, _ = manager.ensure("law")
    assert ev3 == "code"
    assert set(manager._resident) == {"med", "law"}
    again, ev4, load_ms = manager.ensure("code")
    assert ev4 == "med" and load_ms > 0.0
    tol = max(float(subs["code"].values.abs().max()) / 127.0, 1e-6)  # int8-квантование на диске
    assert float((again.values - subs["code"].values).abs().max()) <= tol
    summary = manager.stats_summary()
    assert summary["loads"] == 4 and summary["evictions"] == 2
    assert summary["mean_load_ms"] > 0.0


def test_manager_pin_and_fallback(tmp_path):
    """pin: ручная фиксация тома; без роутера и без pin → деградация к чистому L1."""
    manager, _ = _registered_manager(tmp_path, n_slots=2)
    ids = torch.randint(0, VOCAB, (2, 10), dtype=torch.long)
    features = torch.randn(2, 10, D)
    read = manager.read(features, ids)
    assert read.name is None and read.values.shape[-1] == 0
    manager.pin("code")
    read = manager.read(features, ids)
    assert read.name == "code" and read.weight == 1.0
    assert read.values.shape == (2, 9, D)
    with pytest.raises(KeyError):
        manager.pin("ghost")


# ------------------------------------------------------------------ sidecar


def test_zero_gate_bit_identical_to_baseline(tmp_path):
    """Zero-init γ: attach L2 не меняет выходы ядра бит-в-бит (безопасный attach)."""
    model = _model()
    ids = torch.randint(0, VOCAB, (2, 12), dtype=torch.long)
    baseline_h, baseline_logits = model(ids, torch.zeros(2, 11, D))
    manager = _manager(tmp_path, _fill())
    manager.pin("test-l2")
    sidecar = L2Sidecar.attach(model, manager)
    h, logits = sidecar(ids)
    assert sidecar.gate_norm() == 0.0
    assert torch.equal(h, baseline_h)
    assert torch.equal(logits, baseline_logits)


def test_l2_write_improves_domain_nll(tmp_path):
    """PoC-ядро: Δ-строки «продолжений» снижают NLL на домене, нейтраль не трогают.

    Обучается микро-адаптер инъекции (Proj + γ); ядро и том заморожены —
    маршрут B концепции H-Ngram.
    """
    model = _model()
    facts = _corpus()
    sub = fill_subtable_from_sequences(facts, SPEC, VOCAB)
    manager = _manager(tmp_path, sub)
    manager.pin("test-l2")
    sidecar = L2Sidecar.attach(model, manager)

    neutral = _neutral_seqs(sub)
    nll_before = _nll(sidecar, facts)
    nll_neutral_before = _nll(sidecar, neutral)

    opt = torch.optim.AdamW(sidecar.parameters(), lr=0.1)
    batch = torch.tensor(facts, dtype=torch.long)
    for _ in range(80):
        _, logits = sidecar(batch)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    nll_after = _nll(sidecar, facts)
    nll_neutral_after = _nll(sidecar, neutral)
    assert sidecar.gate_norm() > 0.0
    assert nll_after < nll_before * 0.9, f"domain NLL {nll_before:.4f} → {nll_after:.4f}"
    assert (nll_neutral_after - nll_neutral_before) == 0.0, "нейтраль обязана быть бит-в-бит"


def test_sidecar_requires_aligned_subtable(tmp_path):
    """Субтаблица чужой ширины → явный отказ (не тихий пропуск)."""
    model = _model()
    wide = SubtableSpec(name="wide-l2", n_rows=N_ROWS, row_width=D + 4, key_order=2, hash_scheme="blake2b")
    manager = _manager(tmp_path, fill_subtable_from_sequences(_corpus(2), wide, VOCAB))
    manager.pin("wide-l2")
    sidecar = L2Sidecar.attach(model, manager)
    with pytest.raises(ValueError, match="row_width"):
        sidecar(torch.randint(0, VOCAB, (1, 12), dtype=torch.long))
