"""Дамп held-out eval-документов тренировочного сплита (для написания парафраз).

Воспроизводит сплит из scripts/train_writer.py: корпус Gutenberg, shuffle seed 17,
eval_docs = docs[:8] (held-out — НИКОГДА не записывались в память при обучении).
"""
import random
import urllib.request
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1] / "runs" / "corpus.txt"
if not CORPUS.exists():
    urllib.request.urlretrieve("https://www.gutenberg.org/cache/epub/1661/pg1661.txt", CORPUS)
    print("corpus скачан")

text = CORPUS.read_text(encoding="utf-8-sig")
docs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 300]
rng = random.Random(17)
rng.shuffle(docs)
eval_docs = docs[:8]

for i, doc in enumerate(eval_docs):
    print(f"\n───── eval_doc[{i}] ({len(doc)} chars) ─────")
    print(doc[:600])
