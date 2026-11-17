"""Проверка shipped-чекпоинта (writer_night.safetensors) по методу amargolin (r/LocalLLM):

  1) NLL на held-out документе ДО и ПОСЛЕ записи этого документа (fixed seed);
  2) NLL на hand-written ПАРАФРАЗЕ документа до/после записи оригинала;
  3) NLL на несвязанном held-out тексте до/после записей (должно остаться ≈flat);
Отчёт: /content не нужен — локальный JSON + stdout. Запуск:
  python scripts/eval_checkpoint_nll.py [--writer путь] [--steps N]

Честные рамки: чекпоинт обучен до фикса c_j-семантики; первая запись каждой строки
видит c_j=0 в обоих случаях, так что однократные записи семантически идентичны.
"""
from __future__ import annotations

import json
import random
import sys
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import atomicng
from atomicng.backends.hf import attach_sidecar
from atomicng.writers.io import load_writer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
SPEC = atomicng.TableSpec(name="night-hf-train", n_rows=4096, row_width=16)
CKPT = ROOT.parent / "kaggle-evidence" / "night-sessions" / "atomicng-cpu-night-1" / "writer_night.safetensors"
CORPUS = ROOT / "runs" / "corpus.txt"

# Парафразы eval-документов 0–7 (hand-written, не моделью — чтобы не было циркулярности)
PARAPHRASES = [
    "There was commotion throughout the house after I shouted in anger. Mary came in first, saw the coronet and Arthur's face, understood everything at once, and collapsed unconscious. I sent the maid for the police and handed the case to them immediately. When the inspector and a constable arrived, Arthur stood with folded arms and asked if I meant to accuse him of theft; I replied that it was no longer private but a public affair.",
    "The Project Gutenberg mission promotes the free distribution of electronic works; by using this file you accept the terms of the Full Project Gutenberg License found with the file or online.",
    "Dazed, I walked to the station and asked about the morning train — one to Reading within the hour. The same porter was on duty. He had never heard of Colonel Lysander Stark, had noticed no carriage waiting overnight, and the closest police-station lay three miles off.",
    "Several strokes of my cane struck home and infuriated the creature, which then attacked the first person it saw. In that sense I am indirectly responsible for Dr. Grimesby Roylott's death, though my conscience hardly feels the weight of it.",
    "As long as she remained nearby he could do nothing without a terrible exposure of the woman he loved. The moment she left, he understood the disaster facing you and rushed downstairs in bare feet, opened the window, leapt into the snow and ran down the lane toward a dark figure in the moonlight. Sir George Burnwell tried to flee, but Arthur seized him, and the two struggled over the coronet.",
    "Regarding your health — nothing at all. You know you will soon answer for your deed at a higher court than the Assizes. I shall keep your confession: if McCarthy is condemned I will use it; otherwise it will never meet a human eye, and your secret stays safe with us.",
    "I found him speaking with his son and smoked a cigar behind a tree until they were alone. Listening, I felt all the black bitterness in me rise to the surface: he was urging the boy to marry my daughter.",
    "The stranger produced a photograph of a lady and claimed it was his wife, though she was clearly of a different type. He said he sought amissing partner in a fraudulent scheme involving gold bricks.",
]


def main() -> None:
    writer_ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else CKPT
    report: dict = {"writer": str(writer_ckpt), "model": MODEL_ID, "seed": 17}

    # корпус и тот же сплит, что в тренировке
    if not CORPUS.exists():
        urllib.request.urlretrieve("https://www.gutenberg.org/cache/epub/1661/pg1661.txt", CORPUS)
    text = CORPUS.read_text(encoding="utf-8-sig")
    docs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 300]
    rng = random.Random(17)
    rng.shuffle(docs)
    eval_docs, pool = docs[:8], docs[8:]
    # несвязанный контроль: абзацы, которые НИКОГДА не записывались
    unrelated = pool[-8:]

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    handle = attach_sidecar(model, spec=SPEC, inject_layer=2, seed=17)
    writer, wcfg = load_writer(str(writer_ckpt))
    report["writer_config"] = {k: wcfg[k] for k in ("hidden_size", "row_width", "target_scale", "memory_aware")}

    def nll(ids):
        return round(handle.nll([ids]), 4)

    # 1) acquisition на held-out документах (писали сам документ)
    acq_doc, acq_par, acq_unrel = [], [], []
    for i, doc in enumerate(eval_docs):
        ids = tok.encode(doc, add_special_tokens=False)[:64]
        par_ids = tok.encode(PARAPHRASES[i], add_special_tokens=False)[:64]
        handle.memory.clear()
        before = nll(ids)
        handle.write(ids, writer)
        after = nll(ids)
        par_after = nll(par_ids)
        acq_doc.append(round(before - after, 4))
        acq_par.append(round(before - par_after, 4))
        print(f"doc[{i}]: doc_delta={acq_doc[-1]:+.4f}  paraphrase_delta={acq_par[-1]:+.4f}", flush=True)

    # 2) несвязанный контроль: пишем 8 eval-документов, меряем NLL несвязанных абзацев
    handle.memory.clear()
    unrel_before = [nll([tok.encode(u, add_special_tokens=False)[:64]]) for u in unrelated]
    for doc in eval_docs:
        handle.write(tok.encode(doc, add_special_tokens=False)[:64], writer)
    unrel_after = [nll([tok.encode(u, add_special_tokens=False)[:64]]) for u in unrelated]
    unrel_delta = [round(b - a, 4) for b, a in zip(unrel_before, unrel_after, strict=True)]
    handle.memory.clear()
    handle.detach()

    report.update({
        "acquisition_doc_delta": acq_doc,
        "acquisition_doc_mean": round(sum(acq_doc) / len(acq_doc), 4),
        "paraphrase_delta": acq_par,
        "paraphrase_mean": round(sum(acq_par) / len(acq_par), 4),
        "unrelated_delta_after_writes": unrel_delta,
        "unrelated_mean_abs": round(sum(abs(d) for d in unrel_delta) / len(unrel_delta), 4),
    })
    out = ROOT / "announce" / "checkpoint-eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nREPORT:", json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
