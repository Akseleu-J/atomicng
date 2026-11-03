"""Sidecar-attach к HF-моделям (torch): инжект чтения T⁰+Δ в residual stream.

Поддерживаются causal-LM структуры Llama/Qwen/Mistral-стиля: `model.model.layers[i]`,
`model.model.norm` перед lm_head. Хук на входе слоя `inject_layer` добавляет проекцию
прочитанных строк таблицы; хук на `model.model.norm` захватывает Ĥ (вход lm_head)
для признаков writer'а. Δ хранится в ядре (MemoryPatch, NumPy) — запись only-forward,
в градиенты модели она не вмешивается.

Ограничение v0.2a: sidecar-T⁰ детерминирована от seed (seed сохраняется в provenance
оверлея), перенос оверлея между машинами требует того же seed. Нативные заводские
таблицы — Track B (v1.0).
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
        raise BackendError("backends.hf требует torch: pip install 'atomicng[torch]'") from exc


def bigram_rows(input_ids: list[int], n_rows: int) -> list[int]:
    from .tiny import TinyModel  # noqa: PLC0415

    return [TinyModel.row_hash(input_ids[i - 1], input_ids[i], n_rows) for i in range(1, len(input_ids))]


class SidecarTable:
    """T⁰ (детерминированная от seed, заморожена) + проекция в hidden_size модели."""

    def __init__(self, spec: TableSpec, hidden_size: int, seed: int = 17) -> None:
        torch = _require_torch()
        self.spec = spec
        self.hidden_size = hidden_size
        self.seed = seed
        gen = torch.Generator().manual_seed(seed)
        self.t0 = 0.02 * torch.randn(spec.n_rows, spec.row_width, generator=gen)  # T⁰, иммутабельна
        self.proj = torch.nn.Linear(spec.row_width * spec.n_heads, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.mul_(0.02)  # стартовое влияние памяти мало (ρ-статья)

    def to(self, device: str) -> SidecarTable:
        """Переносит T⁰ и проекцию на устройство (для тренировки на GPU/TPU-VM)."""
        torch = _require_torch()
        self.t0 = self.t0.to(device)
        self.proj = self.proj.to(torch.device(device))
        return self

    def read(self, input_ids: list[int], memory: MemoryPatch) -> np.ndarray:
        """(T, d): строки T⁰+Δ для биграмм последовательности; позиция 0 — нули."""
        t = len(input_ids)
        out = np.zeros((t, self.spec.row_width), dtype=np.float32)
        for i, row in enumerate(bigram_rows(input_ids, self.spec.n_rows), start=1):
            t0_row = self.t0[row]
            if t0_row.is_cuda:
                t0_row = t0_row.detach().cpu()
            out[i] = t0_row.numpy() + memory.get(row)
        return out

    def inject(self, input_ids: list[int], memory: MemoryPatch):
        """→ тензор (T, hidden) для сложения с hidden states слоя (на устройстве proj)."""
        torch = _require_torch()
        read = torch.from_numpy(self.read(input_ids, memory))
        return self.proj(read.to(self.proj.weight.device))


class SidecarHandle:
    """Хуки чтения + запись свидетельств только forward-проходами."""

    def __init__(self, model, spec: TableSpec | None = None, inject_layer: int = 2, seed: int = 17) -> None:
        torch = _require_torch()
        self.torch = torch
        self.spec = spec or SIDECAR_SMALL
        self.inject_layer = inject_layer
        try:
            self.inner = model.model  # LlamaForCausalLM-стиль
            self.layers = model.model.layers
            self.norm = model.model.norm
        except AttributeError as exc:
            raise BackendError(
                "attach_sidecar: не нашёл model.model.layers — поддержаны causal-LM "
                "Llama/Qwen/Mistral-стиля; для своей архитектуры реализуйте хуки вручную"
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

    # ------------------------------------------------------------------ хуки

    def _inject_hook(self, module, args, kwargs):
        h = args[0]
        if self._ids is None:
            return args, kwargs
        mem = self.table.inject(self._ids, self.memory).to(h.dtype).to(h.device)
        if h.dim() == 3:  # (B, T, Hd): одна сессия — батч 1
            mem = mem.unsqueeze(0)
        return (h + mem,) + tuple(args[1:]), kwargs

    def _capture_hook(self, module, args, kwargs):
        h = args[0]
        self.last_hidden = h.detach().to("cpu").float().numpy()

    # --------------------------------------------------------------- сессии

    @contextlib.contextmanager
    def write_session(self, input_ids: list[int]):
        """Задаёт последовательность для чтения таблицы на время одного forward."""
        prev, self._ids = self._ids, list(input_ids)
        try:
            yield self
        finally:
            self._ids = prev

    def forward_hidden(self, input_ids: list[int]):
        """Forward с чтением памяти; возвращает Ĥ (T, Hd) — вход lm_head."""
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
            raise BackendError("модель не вернула hidden_states и хук Ĥ не сработал")
        return self.last_hidden[0]

    # ---------------------------------------------------------------- запись

    def write(self, input_ids: list[int], writer, rule: D1Rule | None = None,
              memory_aware: bool = True) -> dict:
        """Запись свидетельства: Ĥ → writer → цели/гейты → D1 в ядро. Только forward.

        memory_aware=True — delta-вариант статьи (кодирование читает текущую Δ);
        memory_aware=False — plain-вариант: кодирование на T⁰-only модели
        (writer видит нулевые Δ и счётчики), запись в память при этом обычная.
        """
        from ..rules.d1 import Event  # noqa: PLC0415

        hidden = self.forward_hidden(input_ids)          # (T, Hd) — ПОЛНАЯ последовательность
        rows = bigram_rows(input_ids, self.spec.n_rows)  # (T-1,)
        th = self.torch.from_numpy(hidden[None])         # (1, T, Hd): writer сам срежет позицию 0
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
        with self.torch.no_grad():  # запись only-forward: writer не обучается здесь
            targets, gates, _ = writer(th, t_rows, delta_rows, counts)
        events = [
            Event(row_id=rows[i], target=targets[0, i].numpy(), gate=float(gates[0, i]))
            for i in range(len(rows))
        ]
        report = (rule or D1Rule()).apply(self.memory, events)
        self.memory.sources.append(f"hf-write:{len(input_ids)}tok")
        return report.to_dict()

    def nll(self, input_ids_list: list[list[int]]) -> float:
        """Средний NLL (nats/token) последовательностей при текущей памяти T⁰+Δ."""
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

    # --------------------------------------------------------------- оверлеи

    def save_overlay(self, path: str) -> None:
        self.memory.save_pretrained(path)

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def attach_sidecar(model, spec: TableSpec | None = None, inject_layer: int = 2, seed: int = 17) -> SidecarHandle:
    """Подключает sidecar-таблицу к HF-модели (или совместимому стабу) и возвращает handle."""
    return SidecarHandle(model, spec=spec, inject_layer=inject_layer, seed=seed)


def stub_causal_lm(hidden_size: int = 32, vocab_size: int = 64, n_layers: int = 4, seed: int = 3):
    """Мини-стуб causal-LM структуры HF (для тестов attach без скачивания весов)."""
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
