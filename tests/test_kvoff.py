"""KVOff: L0-ярус H-Ngram — KV-кэш, выгружаемый в таблицу для освобождения памяти.

Инварианты PoC (все на настоящем причинном attention с ЯВНЫМ KV-кэшем):

- точная адресация: ключ чувствителен к префиксу (семантика vLLM prefix caching);
- fp32-тир: offload → restore бит-в-бит — генерация с выгрузкой совпадает с чистым
  прогоном (и кэш, и токены);
- recompute-fallback: удалённый с диска блок честно пересчитывается — генерация
  всё равно бит-в-бит (правило отката, «совершенствованнее» наивного offload);
- переиспользование: второй prefill того же промпта — ноль вычислений (restore);
- политика sink+recent: память освобождается, несущие блоки остаются resident;
- int8-тир: компактный, документированная погрешность;
- дистилляция: выгрузка пишет gist-строки в D1-семантику L2-субтаблицы.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomicng.exceptions import OverlayError  # noqa: E402
from atomicng.hngram.kvoff import (  # noqa: E402
    KVBlockSpec,
    KVBlockStore,
    KVOffloadSession,
    KVTinyModel,
    OffloadPolicy,
    block_key,
)
from atomicng.hngram.store import SubtableSpec  # noqa: E402

D, VOCAB = 32, 64
PROMPT = [5, 9, 13, 21, 33, 41, 2, 7, 11, 19, 23, 29]  # 12 токенов = 3 блока по 4


def _model() -> KVTinyModel:
    return KVTinyModel(vocab_size=VOCAB, d=D, n_heads=2, n_layers=2, seed=17)


def _spec(storage: str = "fp32") -> KVBlockSpec:
    return KVBlockSpec(n_layers=2, n_heads=2, head_dim=16, block_len=4, storage=storage)


def _store(tmp_path, storage: str = "fp32", n_slots: int = 4, distill: bool = False) -> KVBlockStore:
    distill_spec = SubtableSpec(name="kv-distill", n_rows=256, row_width=D, key_order=2,
                                hash_scheme="blake2b") if distill else None
    return KVBlockStore(str(tmp_path / "kvstore"), _spec(storage), n_slots=n_slots, distill_spec=distill_spec)


def _generate(session: KVOffloadSession, cache, prompt: list[int], n: int) -> list[int]:
    """Prefill + жадная догенерация n токенов (локальный аналог greedy_generate)."""
    logits, _ = session.prefill(prompt, cache)
    out = [int(logits[0, -1].argmax())]
    for _ in range(n - 1):
        logits = session.forward([out[-1]], cache)
        out.append(int(logits[0, -1].argmax()))
    return out


# ------------------------------------------------------------------ адресация


def test_block_key_exact_and_prefix_sensitive():
    """Ключ точен: цепочечный хеш (vLLM prefix caching) — верифицируем и префикс-чувствителен."""
    assert block_key("m", [1, 2, 3]) == block_key("m", [1, 2, 3])
    assert block_key("m", [1, 2, 3]) != block_key("m", [2, 1, 3])
    # тот же кусок токенов после другого родителя → другой ключ (префикс-чувствительность)
    parent_a, parent_b = block_key("m", [1]), block_key("m", [9])
    assert block_key("m", [2, 3], parent_a) != block_key("m", [2, 3], parent_b)
    # блок верифицируется по (токены, родитель) без хранения всего префикса
    key = block_key("m", [2, 3], parent_a)
    assert key == block_key("m", [2, 3], parent_a)
    assert block_key("m1", [1, 2, 3]) != block_key("m2", [1, 2, 3])


# ------------------------------------------------------------------ хранилище


def test_store_roundtrip_bitexact_and_tamper(tmp_path):
    """fp32-тир: put→get бит-в-бит; tamper → integrity-отказ."""
    store = _store(tmp_path)
    model = _model()
    session = KVOffloadSession(model, store)
    cache = session.new_cache()
    session.prefill(PROMPT, cache)

    keys = [s.key for s in cache.blocks]
    block = store.get(keys[1])
    original = cache.blocks[1]
    assert torch.equal(block.k, original.k) and torch.equal(block.v, original.v)

    path = store._path(keys[2])
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    payload["k"] = payload["k"] * 1.001
    tampered = str(tmp_path / "tampered.npz")
    np.savez(tampered, **payload)
    store._registry[keys[2]] = tampered
    with pytest.raises(OverlayError, match="integrity"):
        store.get(keys[2])


def test_store_int8_mode_error_bound(tmp_path):
    """int8-тир: компактный, ошибка ≤ max_abs/127 (документированная погрешность)."""
    store = _store(tmp_path, storage="int8")
    session = KVOffloadSession(_model(), store)
    cache = session.new_cache()
    session.prefill(PROMPT, cache)
    keys = [s.key for s in cache.blocks]
    block = store.get(keys[0])
    original = cache.blocks[0]
    assert not torch.equal(block.k, original.k)  # lossy
    tol = float(original.k.abs().max()) / 127.0 + 1e-6
    assert float((block.k - original.k).abs().max()) <= tol
    assert float((block.v - original.v).abs().max()) <= float(original.v.abs().max()) / 127.0 + 1e-6


def test_store_lru_eviction_reload_bitexact(tmp_path):
    """LRU-слоты: вытеснение с устройства и перезагрузка не искажают значения."""
    store = _store(tmp_path, n_slots=1)
    session = KVOffloadSession(_model(), store)
    cache = session.new_cache()
    session.prefill(PROMPT, cache)
    keys = [s.key for s in cache.blocks]
    first = store.get(keys[0])
    store.get(keys[1])  # вытесняет first из resident-слотов
    summary = store.stats_summary()
    assert summary["evictions"] >= 1
    again = store.get(keys[0])
    assert torch.equal(again.k, first.k) and torch.equal(again.v, first.v)


# -------------------------------------------------------------- политика


def test_offload_policy_keeps_sink_and_recent(tmp_path):
    """sink+recent остаются resident; середина выгружена, память освобождена."""
    store = _store(tmp_path)
    session = KVOffloadSession(_model(), store)
    cache = session.new_cache()
    session.prefill(PROMPT, cache)  # 3 блока
    assert cache.resident_tokens() == 12
    report = session.offload(cache, OffloadPolicy(sink_blocks=1, recent_blocks=1))
    assert report["freed_tokens"] == 4 and report["resident_tokens"] == 8
    assert cache.blocks[0].resident and cache.blocks[2].resident
    assert not cache.blocks[1].resident
    assert store.has(cache.blocks[1].key)
    assert cache.evicted_slots() == [1]


# ------------------------------------------------- бит-в-бит и fallback


def _continue(session: KVOffloadSession, cache, logits, n: int) -> list[int]:
    """Догенерация n токенов поверх уже наполненного prefill'ом кэша."""
    out = [int(logits[0, -1].argmax())]
    for _ in range(n - 1):
        logits = session.forward([out[-1]], cache)
        out.append(int(logits[0, -1].argmax()))
    return out


def test_generation_bitexact_with_offload_cycle(tmp_path):
    """Ядро PoC: генерация с offload→restore == генерация без выгрузки, бит-в-бит."""
    model = _model()
    pure = KVOffloadSession(model, store=None)
    offl = KVOffloadSession(model, _store(tmp_path))

    cache_pure = pure.new_cache()
    logits_p, _ = pure.prefill(PROMPT, cache_pure)
    gen_pure = _continue(pure, cache_pure, logits_p, 6)

    cache_offl = offl.new_cache()
    logits_o, _ = offl.prefill(PROMPT, cache_offl)
    report = offl.offload(cache_offl, OffloadPolicy(sink_blocks=1, recent_blocks=1))
    assert report["freed_tokens"] > 0
    gen_offl = _continue(offl, cache_offl, logits_o, 6)

    assert gen_pure == gen_offl
    # и сам KV-кэш после цикла выгрузки/восстановления бит-в-бит равен чистому
    # (12 промпт + 5 скормленных forwards; 6-е предсказание не скармливается)
    assert cache_pure.total_tokens() == cache_offl.total_tokens() == 17
    for li in range(model.n_layers):
        kp, vp = cache_pure.layer_kv(li)
        ko, vo = cache_offl.layer_kv(li)
        assert torch.equal(kp, ko) and torch.equal(vp, vo)


def test_prefix_reuse_zero_recompute(tmp_path):
    """Второй prefill того же промпта — все блоки из таблицы, ноль вычислений."""
    model = _model()
    store = _store(tmp_path)
    s1, s2 = KVOffloadSession(model, store), KVOffloadSession(model, store)
    cache1, cache2 = s1.new_cache(), s2.new_cache()
    _, st1 = s1.prefill(PROMPT, cache1)
    assert st1.computed == 3 and st1.stored == 3
    _, st2 = s2.prefill(PROMPT, cache2)
    assert st2.restored == 3 and st2.computed == 0
    # восстановленный кэш бит-в-бит равен вычисленному
    for li in range(model.n_layers):
        k1, v1 = cache1.layer_kv(li)
        k2, v2 = cache2.layer_kv(li)
        assert torch.equal(k1, k2) and torch.equal(v1, v2)


def test_recompute_fallback_after_disk_loss(tmp_path):
    """Точный промах (файл удалён) → пересчёт блока → генерация всё равно бит-в-бит."""
    model = _model()
    pure = KVOffloadSession(model, store=None)
    store = _store(tmp_path)
    offl = KVOffloadSession(model, store)

    cache_pure = pure.new_cache()
    logits_p, _ = pure.prefill(PROMPT, cache_pure)
    gen_pure = _continue(pure, cache_pure, logits_p, 6)

    cache_offl = offl.new_cache()
    logits_o, _ = offl.prefill(PROMPT, cache_offl)
    offl.offload(cache_offl, OffloadPolicy(sink_blocks=1, recent_blocks=1))
    summary_before = store.stats_summary()

    victim = cache_offl.blocks[1].key
    os.remove(store._path(victim))            # «диск потерял блок»
    store._registry.pop(victim)
    gen_offl = _continue(offl, cache_offl, logits_o, 6)  # restore промахнулся → recompute

    assert gen_pure == gen_offl
    assert store.stats_summary()["misses"] > summary_before["misses"]
    # пересчитанный блок вернул кэш в бит-в-бит состояние чистого прогона
    for li in range(model.n_layers):
        kp, vp = cache_pure.layer_kv(li)
        ko, vo = cache_offl.layer_kv(li)
        assert torch.equal(kp, ko) and torch.equal(vp, vo)


# ------------------------------------------------------------- дистилляция


def test_distill_tier_d1_rows(tmp_path):
    """Выгрузка дистиллирует gist-строки в D1-семантике; gist читает «суть» без KV."""
    store = _store(tmp_path, distill=True)
    session = KVOffloadSession(_model(), store)
    cache = session.new_cache()
    session.prefill(PROMPT, cache)
    session.offload(cache)

    table = store.distill_table()
    assert table is not None and bool(table.occupancy.any())
    # хвостовые биграммы всех блоков должны бить в занятые строки
    tails = [PROMPT[w : w + 4][-2:] for w in range(0, 12, 4)]
    from atomicng.hngram.hashing import blake_addresses

    rows = [blake_addresses(t, 256, 2)[-1] for t in tails]
    vals, hits = table.lookup(torch.tensor([rows]))
    assert hits.all() and float(vals[0].norm()) > 0
    gist = store.gist(PROMPT[-2:])  # та же биграмма-хвост — суть доступна без KV
    assert gist is not None and float(gist.norm()) > 0
