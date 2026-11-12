"""AtomicNG Writer Training on a Real Model and Body (mini-CPU v0.3.1)).

Example:
    python scripts/train_writer.py --model HuggingFaceTB/SmolLM2-135M \
        --corpus runs/corpus.txt --out runs/demo --steps 30

Pipeline: body → paragraphs → blocks (query/continuation) → BPTT training HFWriterTrainer
→ checkpoint safetensors → eval acquisition on withheld documents (Δ)_empty per doc).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from atomicng.backends.hf import SidecarTable, attach_sidecar
from atomicng.evidence import build_blocks
from atomicng.specs import TableSpec
from atomicng.writers.hf_training import Block, HFWriterTrainer
from atomicng.writers.io import save_writer, writer_fingerprint
from atomicng.writers.mlp import WriterMLP


def load_corpus(path: str, min_chars: int = 300) -> list[str]:
    text = Path(path).read_text(encoding="utf-8-sig")
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_chars]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train_writer")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--trajectory", type=int, default=3, help="block-track")
    ap.add_argument("--block", type=int, default=96, help="block-token")
    ap.add_argument("--query", type=int, default=24, help="query-block prefix")
    ap.add_argument("--row-width", type=int, default=16)
    ap.add_argument("--n-rows", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-docs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    docs = load_corpus(args.corpus)
    if len(docs) < args.eval_docs + 4:
        raise SystemExit(f"hull: {len(docs)} paragraph")
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    eval_docs, train_docs = docs[: args.eval_docs], docs[args.eval_docs :]
    print(f"corpus: {len(docs)} paragraph (train) {len(train_docs)}, eval {len(eval_docs)})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    hidden_size = int(model.config.hidden_size)

    spec = TableSpec(
        name=f"sidecar-{args.model.split('/')[-1]}", n_rows=args.n_rows, row_width=args.row_width
    )
    table = SidecarTable(spec, hidden_size, seed=args.seed)
    writer = WriterMLP(hidden_size=hidden_size, row_width=args.row_width, hidden=64, target_scale=0.3)
    trainer = HFWriterTrainer(model, writer, spec, table, lr=args.lr)
    print(f"model {args.model}: hidden={hidden_size}, inject_layer={trainer.inject_layer}", flush=True)

    def doc_blocks(doc: str) -> list[Block]:
        ids = tok.encode(doc, add_special_tokens=False)
        return [
            Block(b.tokens, len(b.query))
            for b in build_blocks(ids, block_size=args.block, query_prefix=args.query)
            if len(b.tokens) > args.query + 1
        ]

    all_blocks: list[Block] = []
    for doc in train_docs:
        all_blocks.extend(doc_blocks(doc))
    if not all_blocks:
        raise SystemExit("no blocks after slicing")
    trajectories = [
        all_blocks[i : i + args.trajectory]
        for i in range(0, len(all_blocks) - args.trajectory + 1, args.trajectory)
    ]
    print(f"block: {len(all_blocks)}, trajectory: {len(trajectories)}", flush=True)

    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        traj = trajectories[step % len(trajectories)]
        loss = trainer.train_group([traj])
        losses.append(round(loss, 4))
        if step % 5 == 0 or step == args.steps - 1:
            print(f"step {step}: loss {loss:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    trainer.close()
    writer.eval()

    ckpt = out / "writer.safetensors"
    corpus_sha = hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()
    save_writer(
        writer,
        str(ckpt),
        extra={
            "model_id": args.model,
            "steps": args.steps,
            "spec_name": spec.name,
            "corpus_sha256": corpus_sha,
            "block": args.block,
            "query": args.query,
        },
    )
    from atomicng.writers.io import save_sidecar_proj
    save_sidecar_proj(table, str(out / "sidecar_proj.safetensors"), seed=args.seed)
    print("sidecar proj saved:", out / "sidecar_proj.safetensors", flush=True)

    # eval acquisition: withheld documents, empty memory vs memory of the document itself
    handle = attach_sidecar(model, spec=spec, inject_layer=trainer.inject_layer, seed=args.seed)
    handle.table = table  # same T⁰/proj, what
    deltas: list[float] = []
    for doc in eval_docs:
        ids = tok.encode(doc, add_special_tokens=False)[: args.block]
        if len(ids) < 8:
            continue
        handle.memory.clear()
        nll_empty = handle.nll([ids])
        handle.write(ids, writer)
        deltas.append(nll_empty - handle.nll([ids]))
    handle.memory.clear()
    handle.detach()

    report = {
        "model": args.model,
        "steps": args.steps,
        "loss_first5": sum(losses[:5]) / max(len(losses[:5]), 1),
        "loss_last5": sum(losses[-5:]) / max(len(losses[-5:]), 1),
        "writer_fingerprint": writer_fingerprint(writer),
        "checkpoint": str(ckpt),
        "acquisition": {
            "n_docs": len(deltas),
            "delta_empty_mean": round(sum(deltas) / max(len(deltas), 1), 4),
            "delta_empty_per_doc": [round(d, 4) for d in deltas],
        },
    }
    (out / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
