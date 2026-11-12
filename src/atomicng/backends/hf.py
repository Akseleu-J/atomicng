"""PyTorch sidecar for Hugging Face causal language models.

A pre-hook at the selected transformer layer injects projected T⁰+Δ reads
into the residual stream. A hook on model.model.norm captures Ĥ before
lm_head. The NumPy MemoryPatch stores forward-only writes. The sidecar's
base table is generated from a seed recorded in overlay provenance.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np

from ..exceptions import BackendError
from ..memory import MemoryPatch
from ..rules.d1 import D1Rule
from ..specs import SIDECAR_SMALL, TableSpec


def _require_torch():
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError as exc:
        raise BackendError("backends.hf requires torch: pip install 'atomicng[torch]'") from exc


def bigram_rows(input_ids: list[int], n_rows: int) -> list[int]:
    from .tiny import TinyModel  # noqa: PLC0415

    return [TinyModel.row_hash(input_ids[i - 1], input_ids[i], n_rows) for i in range(1, len(input_ids))]


class SidecarTable:
    """T⁰ (frozen and generated from a fixed seed) + projection_size model."""

    def __init__(self, spec: TableSpec, hidden_size: int, seed: int = 17) -> None:
        torch = _require_torch()
        self.spec = spec
        self.hidden_size = hidden_size
        self.seed = seed
        gen = torch.Generator().manual_seed(seed)
        self.t0 = 0.02 * torch.randn(spec.n_rows, spec.row_width, generator=gen)  # T⁰, immutable
        self.proj = torch.nn.Linear(spec.row_width * spec.n_heads, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.mul_(0.02)  # The initial effect of memory is small (ρ-article))

    def to(self, device: str) -> SidecarTable:
        """Move T⁰ and the projection to the training device."""
        torch = _require_torch()
        self.t0 = self.t0.to(device)
        self.proj = self.proj.to(torch.device(device))
        return self

    def read(self, input_ids: list[int], memory: MemoryPatch) -> np.ndarray:
        """Return T⁰+Δ rows for sequence bigrams as a (T, d) tensor. Position 0 is zero."""
        t = len(input_ids)
        out = np.zeros((t, self.spec.row_width), dtype=np.float32)
        for i, row in enumerate(bigram_rows(input_ids, self.spec.n_rows), start=1):
            t0_row = self.t0[row]
            if t0_row.is_cuda:
                t0_row = t0_row.detach().cpu()
            out[i] = t0_row.numpy() + memory.get(row)
        return out

    def inject(self, input_ids: list[int], memory: MemoryPatch):
        """→ tensor (T, hidden) for addition to the hidden states layer (on the device proj))."""
        torch = _require_torch()
        read = torch.from_numpy(self.read(input_ids, memory))
        return self.proj(read.to(self.proj.weight.device))


class SidecarHandle:
    """Hook reading + Recording of certificates only forward passes."""

    def __init__(self, model, spec: TableSpec | None = None, inject_layer: int = 2, seed: int = 17) -> None:
        torch = _require_torch()
        self.torch = torch
        self.spec = spec or SIDECAR_SMALL
        self.inject_layer = inject_layer
        try:
            self.inner = model.model  # LlamaForCausalLM-style
            self.layers = model.model.layers
            self.norm = model.model.norm
        except AttributeError as exc:
            raise BackendError(
                "attach_sidecar: No model.model.layers supported by causal-LM "
                "of the Llama, Qwen, or Mistral family; For your architecture, implement hooks manually."
            ) from exc
        hidden_size = int(model.config.hidden_size)
        self.table = SidecarTable(self.spec, hidden_size, seed=seed)
        self.memory = MemoryPatch(self.spec, created_by=f"hf-sidecar:seed={seed}")
        self.model_ref = model
        self._ids: list[int] | None = None
        self.last_hidden: np.ndarray | None = None
        self._hooks = [
            self.layers[inject_layer].register_forward_pre_hook(self._inject_hook, with_kwargs=True),
            self.norm.register_forward_pre_hook(self._capture_hook, with_kwargs=True),
        ]

    # ------------------------------------------------------------------ hook

    def _inject_hook(self, module, args, kwargs):
        h = args[0]
        if self._ids is None:
            return args, kwargs
        mem = self.table.inject(self._ids, self.memory).to(h.dtype).to(h.device)
        if h.dim() == 3:  # (B, T, Hd: one session - batch 1
            mem = mem.unsqueeze(0)
        return (h + mem,) + tuple(args[1:]), kwargs

    def _capture_hook(self, module, args, kwargs):
        h = args[0]
        self.last_hidden = h.detach().to("cpu").float().numpy()

    # --------------------------------------------------------------- session

    @contextlib.contextmanager
    def write_session(self, input_ids: list[int]):
        """Set the sequence for reading a table for one time forward."""
        prev, self._ids = self._ids, list(input_ids)
        try:
            yield self
        finally:
            self._ids = prev

    def forward_hidden(self, input_ids: list[int]):
        """Run a forward pass with memory and return Ĥ (T, Hd), the input to lm_head."""
        torch = self.torch
        was_training = self.model_ref.training
        self.model_ref.eval()
        dev = next(self.model_ref.parameters()).device
        with torch.no_grad():
            with self.write_session(input_ids):
                outputs = self.model_ref(input_ids=torch.tensor([input_ids], dtype=torch.long, device=dev))
        if was_training:
            self.model_ref.train()
        if outputs.hidden_states is not None:
            return outputs.hidden_states[-1][0].numpy()
        if self.last_hidden is None:
            raise BackendError("The model returned no hidden_states, and the Ĥ hook did not fire.")
        return self.last_hidden[0]

    # ---------------------------------------------------------------- record

    def write(self, input_ids: list[int], writer, rule: D1Rule | None = None,
              memory_aware: bool = True) -> dict:
        """Write evidence with a forward pass, then compute targets and gates from Ĥ and apply D1.

        memory_aware=True is a delta version of the article (coding reads the current Δ));
        Set memory_aware=False for plain encoding using the base table alone
        (The writer sees zero Δ and counters, and the memory recording is normal..
        """
        from ..rules.d1 import Event  # noqa: PLC0415

        hidden = self.forward_hidden(input_ids)          # (T, Hd) - Complete sequence
        rows = bigram_rows(input_ids, self.spec.n_rows)  # (T-1,)
        th = self.torch.from_numpy(hidden[None])         # (1, T, Hd: writer cuts the position 0
        t_rows = self.torch.tensor([rows], dtype=self.torch.long)
        if memory_aware:
            delta_rows = self.torch.stack(
                [self.torch.from_numpy(self.memory.get(r)) for r in rows]
            )[None]
            counts = self.torch.tensor(
                [[float(self.memory.count(r)) for r in rows]], dtype=self.torch.float32
            )
        else:
            zero = self.torch.zeros(1, len(rows), self.spec.row_width)
            delta_rows, counts = zero, self.torch.zeros(1, len(rows))
        with self.torch.no_grad():  # Only-forward: The writer is not trained here
            targets, gates, _ = writer(th, t_rows, delta_rows, counts)
        events = [
            Event(row_id=rows[i], target=targets[0, i].numpy(), gate=float(gates[0, i]))
            for i in range(len(rows))
        ]
        report = (rule or D1Rule()).apply(self.memory, events)
        self.memory.sources.append(f"hf-write:{len(input_ids)}tok")
        return report.to_dict()

    def nll(self, input_ids_list: list[list[int]]) -> float:
        """Average NLL (nats/token) for the sequences using the current memory T⁰+Δ."""
        total, count = 0.0, 0
        for ids in input_ids_list:
            if len(ids) < 2:
                continue
            with self.write_session(ids), self.torch.no_grad():
                dev = next(self.model_ref.parameters()).device
                out = self.model_ref(input_ids=self.torch.tensor([ids], dtype=self.torch.long, device=dev))
            logits = out.logits[0].float()
            logprobs = self.torch.log_softmax(logits[:-1], dim=-1)
            targets = self.torch.tensor(ids[1:], dtype=self.torch.long, device=logits.device)
            total -= float(logprobs[self.torch.arange(len(ids) - 1, device=logits.device), targets].sum())
            count += len(ids) - 1
        return total / max(count, 1)

    # --------------------------------------------------------------- overlay

    def save_overlay(self, path: str) -> None:
        self.memory.save_pretrained(path)

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def attach_sidecar(model, spec: TableSpec | None = None, inject_layer: int = 2, seed: int = 17) -> SidecarHandle:
    """Attach the sidecar to a compatible Hugging Face model and return its handle."""
    return SidecarHandle(model, spec=spec, inject_layer=inject_layer, seed=seed)


def stub_causal_lm(hidden_size: int = 32, vocab_size: int = 64, n_layers: int = 4, seed: int = 3):
    """Mini-stood causal-LM structure HF (for attach tests without downloading weights))."""
    torch = _require_torch()
    gen = torch.Generator().manual_seed(seed)

    class StubLayer(torch.nn.Module):
        def forward(self, h):
            return h + 0.1 * torch.tanh(h)

    class StubInner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(StubLayer() for _ in range(n_layers))
            self.norm = torch.nn.RMSNorm(hidden_size)

    class StubModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)
            self.embed = torch.nn.Parameter(0.1 * torch.randn(vocab_size, hidden_size, generator=gen))
            self.model = StubInner()
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

        def forward(self, input_ids, **_):
            h = self.embed[input_ids]
            for layer in self.model.layers:
                h = layer(h)
            h = self.model.norm(h)
            return SimpleNamespace(logits=self.lm_head(h), hidden_states=None)

    return StubModel()
