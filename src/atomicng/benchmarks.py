"""Реестр протоколов оценки статьи (T5): acquisition, distraction, chain.

Все протоколы бэкенд-агностичны: модели нужен метод `nll(seqs) → nats/token`
(реализован у TinyModel; для HF-моделей — адаптер с логитами), writer'у — `events(texts)`.
Каждый протокол снапшотит патч в начале и восстанавливает его в конце — состояние
модели не меняется (P0).

Интерпретация (как в статье):
- `delta_empty` > 0 — память с записью документа снижает NLL его продолжений относительно
  пустой памяти (приобретение);
- `delta_donor` > `delta_empty` — память документа специфична (чужая память помогает хуже);
- `retention@k` — доля приобретения, сохранившаяся после k отвлекающих записей
  (в статье plain 54%, delta 70% при k=16);
- `drift` — сдвиг NLL на нейтральных пробах по мере цепочки записей (big-K drift);
- `recall` — сохранение приобретения ранее записанных документов (small-k).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .rules.d1 import D1Rule
from .specs import TableSpec
from .testing import HashWriter


def _seqs(model, docs: Sequence[str], tokenizer=None) -> list[list[int]]:
    if tokenizer is not None:
        return [tokenizer.encode(d) for d in docs]
    writer = HashWriter(model.spec)
    return [writer.tokenize(d) for d in docs]


def _write_all(model, docs: Sequence[str], writer) -> None:
    for doc in docs:
        model.memory.bump_clock()
        D1Rule().apply(model.memory, writer.events([doc]))


def acquisition(
    model,
    docs: Sequence[str],
    *,
    writer=None,
    tokenizer=None,
) -> dict:
    """Δ_empty / Δ_donor: пользу памяти документа против пустой и чужой памяти."""
    writer = writer or HashWriter(model.spec)
    seqs = [s for s in _seqs(model, docs, tokenizer) if len(s) >= 2]
    if len(docs) < 2:
        raise ValueError("acquisition требует ≥ 2 документов (нужен донор)")

    snap = model.memory.snapshot()
    try:
        model.memory.clear()
        nll_empty = model.nll(seqs)

        _write_all(model, docs, writer)
        nll_self = model.nll(seqs)

        model.memory.clear()
        _write_all(model, docs[1:], writer)  # донор: всё, кроме целевого документа
        nll_donor = model.nll(seqs)
    finally:
        model.memory.restore(snap)

    # значения БЕЗ округления: тесты сверяют арифметику дельт бит-в-бит на любой платформе
    return {
        "n_docs": len(docs),
        "nll_empty": nll_empty,
        "nll_self": nll_self,
        "nll_donor": nll_donor,
        "delta_empty": nll_empty - nll_self,
        "delta_donor": nll_donor - nll_self,
    }


def distraction(
    model,
    docs: Sequence[str],
    *,
    ks: Sequence[int] = (0, 1, 4, 8, 16),
    writer=None,
    tokenizer=None,
) -> dict:
    """Retention@k: доля приобретения после k отвлекающих записей поверх целевого документа."""
    writer = writer or HashWriter(model.spec)
    if len(docs) < 2:
        raise ValueError("distraction требует целевой документ и отвлекающие")
    target = [_seqs(model, docs[:1], tokenizer)[0]]
    if len(target[0]) < 2:
        raise ValueError("целевой документ короче 2 токенов")
    distractors = list(docs[1:])

    snap = model.memory.snapshot()
    try:
        model.memory.clear()
        nll_empty = model.nll(target)

        _write_all(model, docs[:1], writer)
        gains: dict[int, float] = {}
        nll0 = model.nll(target)
        base_gain = nll_empty - nll0
        gains[0] = round(base_gain, 6)

        written = 0
        for k in sorted(ks):
            if k == 0:
                continue
            while written < k and written < len(distractors):
                _write_all(model, distractors[written : written + 1], writer)
                written += 1
            gains[k] = round(nll_empty - model.nll(target), 6)

        retention = {
            k: (round(gains[k] / base_gain, 4) if base_gain > 0 else None) for k in sorted(ks)
        }
    finally:
        model.memory.restore(snap)

    return {
        "n_distractors_available": len(distractors),
        "gain_nats": gains,
        "retention_at_k": retention,
    }


def chain(
    model,
    docs: Sequence[str],
    *,
    writer=None,
    tokenizer=None,
    n_probes: int = 32,
) -> dict:
    """Цепочка записей: дрейф на нейтральных пробах и recall ранее записанного.

    Пробы фиксируются ДО первой записи; коллизии с новыми строками — честный дрейф
    (в статье big-K drift растёт именно так). Состояние модели восстанавливается.
    """
    writer = writer or HashWriter(model.spec)
    doc_seqs = [s for s in _seqs(model, docs, tokenizer) if len(s) >= 2]
    if not doc_seqs:
        raise ValueError("chain: нет документов длиннее 2 токенов")

    snap = model.memory.snapshot()
    try:
        model.memory.clear()
        probes = model.neutral_token_seqs(n_probes) if hasattr(model, "neutral_token_seqs") else None
        nll_neutral_base = model.nll(probes) if probes else float("nan")
        pairs = [(docs[i], s) for i, s in enumerate(_seqs(model, docs, tokenizer)) if len(s) >= 2]
        nll_docs_base = [model.nll([s]) for _, s in pairs]

        drift_curve: list[float] = []
        recall_curve: list[float] = []
        immediate_gains: list[float] = []
        for i, (doc, _seq) in enumerate(pairs):
            _write_all(model, [doc], writer)
            drift_curve.append(round(model.nll(probes) - nll_neutral_base, 6) if probes else float("nan"))
            # recall: приобретение всех уже записанных документов относительно пустой памяти
            gains = [nll_docs_base[j] - model.nll([pairs[j][1]]) for j in range(i + 1)]
            recall_curve.append(round(float(np.mean(gains)), 6))
            immediate_gains.append(gains[-1])

        stability = (
            round(float(np.mean(recall_curve[-1])) / float(np.mean(immediate_gains)), 4)
            if np.mean(immediate_gains) > 0
            else None
        )
    finally:
        model.memory.restore(snap)

    return {
        "n_steps": len(doc_seqs),
        "nll_neutral_base": round(nll_neutral_base, 6),
        "drift_curve": drift_curve,
        "recall_curve": recall_curve,
        "final_drift": drift_curve[-1],
        "final_recall": recall_curve[-1],
        "stability": stability,
    }


def run_suite(
    model,
    docs: Sequence[str],
    *,
    writer=None,
    tokenizer=None,
    chain_steps: int | None = None,
) -> dict:
    """Полный набор протоколов над одним корпусом (для CLI `atomicng eval`)."""
    docs = list(docs)
    if len(docs) < 2:
        raise ValueError("run_suite требует ≥ 2 документов")
    return {
        "acquisition": acquisition(model, docs, writer=writer, tokenizer=tokenizer),
        "distraction": distraction(
            model,
            docs,
            ks=tuple(k for k in (0, 1, 4, 8, 16) if k < len(docs)),
            writer=writer,
            tokenizer=tokenizer,
        ),
        "chain": chain(model, docs[: chain_steps] if chain_steps else docs, writer=writer, tokenizer=tokenizer),
    }


def spec_of(model) -> TableSpec:
    """Спека модели (удобство для CLI/адаптеров)."""
    spec = getattr(model, "spec", None)
    if not isinstance(spec, TableSpec):
        raise ValueError("у модели нет .spec (TableSpec)")
    return spec
