"""Обучение writer'а на настоящей HF-модели (Алгоритм 2 статьи, реальный путь).

Отличие от stands (TorchTinyModel): модель — настоящий causal-LM из transformers;
Δ входит в forward через дифференцируемый инжект перед слоем L (proj заморожен,
но градиент течёт СКВОЗЬ него в Δ и дальше в θ writer'а); loss — CE на продолжении
блока (query-префикс предсказывается, потом блок записывается — протокол статьи);
один backward на группу траекторий; память стартует пустой на каждую траекторию.

Требования к модели: model.model.layers[...] и model.model.norm (Llama/Qwen/Mistral-стиль).
"""
from __future__ import annotations

import torch
from torch import nn

from ..backends.hf import SidecarTable, bigram_rows
from ..exceptions import BackendError
from ..specs import TableSpec


class Block:
    """Блок обучения: полная последовательность id + длина query-префикса."""

    __slots__ = ("ids", "query_len")

    def __init__(self, ids: list[int], query_len: int) -> None:
        if not 0 <= query_len < len(ids):
            raise ValueError(f"query_len={query_len} вне диапазона длины {len(ids)}")
        self.ids = ids
        self.query_len = query_len

    @property
    def tokens(self) -> list[int]:
        return self.ids


class HFWriterTrainer:
    """BPTT-тренер writer'а поверх настоящей HF-модели (свои хуки, модель заморожена)."""

    def __init__(
        self,
        model,
        writer: nn.Module,
        spec: TableSpec,
        table: SidecarTable,
        *,
        inject_layer: int | None = None,
        lr: float = 3e-4,
        clip: float = 1.0,
        device: str = "cpu",
    ) -> None:
        try:
            self.layers = model.model.layers
            self.norm = model.model.norm
        except AttributeError as exc:
            raise BackendError(
                "HFWriterTrainer: нужна модель с model.model.layers / model.model.norm"
            ) from exc
        self.device = device
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in table.proj.parameters():
            p.requires_grad_(False)  # proj заморожен, но градиент течёт СКВОЗЬ него в Δ
        self.writer = writer.to(device).train()
        self.spec = spec
        self.table = table.to(device)
        self.inject_layer = min(inject_layer if inject_layer is not None else 2, len(self.layers) - 1)
        self.opt = torch.optim.AdamW(self.writer.parameters(), lr=lr, weight_decay=0.0)
        self.clip = clip
        # состояние текущей траектории (сбрасывается на каждую)
        self._delta: dict[int, torch.Tensor] = {}
        self._counts: dict[int, int] = {}
        self._ids: list[int] | None = None
        self._hidden: torch.Tensor | None = None
        self._hooks = [
            self.layers[self.inject_layer].register_forward_pre_hook(self._inject_hook, with_kwargs=True),
            self.norm.register_forward_pre_hook(self._capture_hook, with_kwargs=True),
        ]

    # ------------------------------------------------------------------ хуки

    def _inject_hook(self, module, args, kwargs):
        h = args[0]
        if self._ids is None:
            return args, kwargs
        mem = self._mem_seq(self._ids).to(h.dtype)
        if h.dim() == 3:
            mem = mem.unsqueeze(0)
        return (h + mem,) + tuple(args[1:]), kwargs

    def _mem_seq(self, ids: list[int]) -> torch.Tensor:
        """(T, hidden): чтение T⁰+Δ по биграммам + проекция; градиент течёт через Δ."""
        rows = bigram_rows(ids, self.spec.n_rows)
        t0 = self.table.t0
        zero = torch.zeros(self.spec.row_width, device=t0.device)
        read = torch.stack([t0[r] + self._delta.get(r, zero) for r in rows])
        pad = torch.zeros(1, self.spec.row_width, device=t0.device)
        return self.table.proj(torch.cat([pad, read], dim=0))

    def _capture_hook(self, module, args, kwargs):
        self._hidden = args[0].detach()  # Ĥ для признаков writer'а (grad-stop)

    # -------------------------------------------------------------- обучение

    def train_group(self, trajectories: list[list[Block]]) -> float:
        """Один апдейт на группу траекторий; возвращает средний CE продолжений (nats/token)."""
        self.opt.zero_grad(set_to_none=True)
        group_terms: list[torch.Tensor] = []            # mean_ce блока / len(traj) — нормировка группы
        report_terms: list[tuple[torch.Tensor, int]] = []  # (mean_ce блока, n токенов) — для отчёта
        n_tokens = 0
        for traj in trajectories:
            self._delta = {}
            self._counts = {}
            for block in traj:
                self._ids = block.ids
                out = self.model(input_ids=torch.tensor([block.ids], dtype=torch.long, device=self.device))
                logits = out.logits[0].float()          # (T, V); градиент через Δ-инжект
                start, end = block.query_len - 1, len(block.ids) - 1
                if end > start:
                    logprobs = torch.log_softmax(logits[start:end], dim=-1)
                    targets = torch.tensor(block.ids[start + 1 : end + 1], dtype=torch.long, device=logits.device)
                    mean_ce = -logprobs[torch.arange(end - start, device=logits.device), targets].mean()
                    group_terms.append(mean_ce / len(traj))
                    report_terms.append((mean_ce, end - start))
                    n_tokens += end - start

                self._write_block(block.ids)
        self._ids = None
        if not group_terms:
            return float("nan")
        report_mean = sum(loss * n for loss, n in report_terms) / max(n_tokens, 1)
        total = sum(group_terms)
        if not total.requires_grad:
            # Группа без градиентного пути (например, один блок: память пишется ПОСЛЕ
            # предсказания) — обновление имеет смысл только при наличии цепочки записей.
            return float(report_mean.detach())
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.writer.parameters(), self.clip)
        self.opt.step()
        return float(report_mean.detach())

    def _write_block(self, ids: list[int]) -> None:
        """Дифференцируемая запись блока: D1 block averaging в δ-словарь траектории."""
        rows = bigram_rows(ids, self.spec.n_rows)
        hidden = self._hidden
        if hidden is None:
            raise BackendError("Ĥ не захвачен: хук на model.model.norm не сработал")
        # хук norm даёт (B, T, Hd) — для батча 1 это уже (1, T, Hd); writer сам срежет позицию 0
        hd = hidden if hidden.dim() == 3 else hidden.unsqueeze(0)
        dev = hidden.device
        t_rows = torch.tensor([rows], dtype=torch.long, device=dev)
        zero_d = torch.zeros(self.spec.row_width, device=dev)
        delta_rows = torch.stack([self._delta.get(r, zero_d) for r in rows]).unsqueeze(0)
        counts_t = torch.tensor(
            [[float(self._counts.get(r, 0)) for r in rows]], dtype=torch.float32, device=dev
        )  # уже (1, T-1)
        # hd уже detached (grad-stop на Ĥ); writer вызывается С ГРАДИЕНТОМ —
        # targets/gates несут BPTT: loss блока k+1 течёт через Δ в θ writer'а
        targets, gates, _ = self.writer(hd, t_rows, delta_rows, counts_t)
        flat_rows = rows
        flat_t = targets.reshape(-1, self.spec.row_width)
        flat_g = gates.reshape(-1)
        for r in sorted(set(rows)):
            mask = torch.tensor([x == r for x in flat_rows], device=dev)
            a = (flat_g[mask][:, None] * flat_t[mask]).mean(dim=0)
            w = flat_g[mask].mean()
            old = self._delta.get(r, torch.zeros(self.spec.row_width, device=dev))
            self._delta[r] = (1.0 - w) * old + a
            self._counts[r] = self._counts.get(r, 0) + int(mask.sum())

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
