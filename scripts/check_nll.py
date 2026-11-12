"""Quick check: basic NLL SmolLM2 on the corpus paragraph (diagnostic loss in the coach))."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch
import torch.nn.functional as F
from train_writer import load_corpus
from transformers import AutoModelForCausalLM, AutoTokenizer

docs = load_corpus(str(Path(__file__).resolve().parents[1] / "runs" / "corpus.txt"))
print("doc0 head:", docs[0][:120].replace("\n", " "))
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M").eval()
ids = tok.encode(docs[0], add_special_tokens=False)[:96]
with torch.no_grad():
    logits = model(input_ids=torch.tensor([ids])).logits[0].float()
logprobs = F.log_softmax(logits[:-1], dim=-1)
targets = torch.tensor(ids[1:])
per_token = -logprobs[torch.arange(len(targets)), targets]
print(f"plain NLL/token: {float(per_token.mean()):.4f} (min {float(per_token.min()):.3f}, max {float(per_token.max()):.3f})")
print("id sample:", ids[:20])
