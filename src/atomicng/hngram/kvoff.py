"""KVOff — L0-ярус иерархии H-Ngram: KV-кэш, выгружаемый в таблицу для освобождения памяти.

Механизм зеркалит AtomicNG и расширяет его:

- **Точная адресация.** Ключ блока = sha256(model_tag + ВСЕХ токенов до конца блока)
  (семантика vLLM prefix caching: KV блока зависит от всего префикса). Хеш НЕ по
  модулю: коллизия KV-адреса — молчаливая порча генерации, поэтому внутри
  хранилища ключ сравнивается точно; poly64/blake остаются на ярусах знаний.
- **LRU-слоты на устройстве** — как SubtableManager: resident-блоки на девайсе,
  остальные на «диске» (npz + sha256-манифест, конвенции overlay).
- **Два режима хранения:** fp32 — бит-в-бит (для точного восстановления), int8 —
  компактный (документированная погрешность, как квантование KV в проде).
- **Offload-политика** (StreamingLLM/H2O-стиль): sink-блоки и свежее окно остаются
  resident, середина выгружается → память освобождается.
- **Recompute-fallback** («откат к чистому ядру»): точный промах при restore не
  роняет систему — блок честно пересчитывается из токенов поверх resident-префикса.
- **Дистилляция в D1-строки** (мост к L2-субтаблице): при выгрузке блок пишет в
  n-gram таблицу («gist»: биграмма-хвост блока → средний V блока, D1-блок-усреднение).
  Точный ярус — восстановление; дистиллят — знания без точного восстановления.

PoC-ядро — KVTinyModel: крошечный причинный трансформер с ЯВНЫМ KV-кэшем (frozen,
seeded), чтобы цикл prefill → offload → restore → generate проверялся бит-в-бит.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from ..exceptions import OverlayError
from ..rules.d1 import D1Rule, Event
from .store import L2Subtable, SubtableSpec

KVOFF_FORMAT = "atomicng-hngram-kvoff"
KVOFF_VERSION = 1
STORAGE_MODES = ("fp32", "int8")


def block_key(model_tag: str, tokens: list[int], parent_key: str | None = None) -> str:
    """Точный адрес KV-блока: цепочечный sha256 (схема vLLM prefix caching).

    key_i = sha256(model_tag | parent_key | tokens_i), где parent_key — ключ
    предыдущего блока. Префикс-чувствительность через цепочку: тот же кусок
    токенов после другого контекста даёт другой ключ; при этом блок верифицируется
    по (токены, parent_key) без хранения всего префикса в записи.
    """
    payload = model_tag + "|" + (parent_key or "") + "|" + ",".join(str(int(t)) for t in tokens)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class KVBlockSpec:
    """Геометрия KV-блока и режим хранения."""

    n_layers: int = 2
    n_heads: int = 2
    head_dim: int = 16
    block_len: int = 4
    storage: str = "fp32"  # "fp32" — lossless (бит-в-бит), "int8" — компактно

    def __post_init__(self) -> None:
        if min(self.n_layers, self.n_heads, self.head_dim, self.block_len) < 1:
            raise ValueError("All KVBlockSpec dimensions must be positive.")
        if self.storage not in STORAGE_MODES:
            raise ValueError(f"storage must be one of {STORAGE_MODES}.")


@dataclass
class KVBlock:
    """KV-блок: кусок токенов + K/V всех слоёв + ключ родителя (цепочка префикса)."""

    tokens: list[int]
    key: str                      # точный адрес (цепочечный prefix-hash)
    k: torch.Tensor
    v: torch.Tensor
    parent_key: str | None = None

    @property
    def length(self) -> int:
        return len(self.tokens)


# ----------------------------------------------------------------- KV-кэш


@dataclass
class _Slot:
    """Слот пейджированного кэша: resident (k,v) или вытесненная заглушка."""

    tokens: list[int]
    key: str
    parent_key: str | None = None
    k: torch.Tensor | None = None  # (n_layers, n_heads, len, head_dim)
    v: torch.Tensor | None = None

    @property
    def resident(self) -> bool:
        return self.k is not None


class KVCache:
    """Пейджированный KV-кэш: список слотов-блоков, evict освобождает тензоры."""

    def __init__(self, spec: KVBlockSpec) -> None:
        self.spec = spec
        self.blocks: list[_Slot] = []

    def append_block(self, tokens: list[int], key: str, k: torch.Tensor, v: torch.Tensor,
                     parent_key: str | None = None) -> None:
        if k.shape != (self.spec.n_layers, self.spec.n_heads, len(tokens), self.spec.head_dim):
            raise OverlayError(f"KV block shape {tuple(k.shape)} does not match spec {self.spec}.")
        self.blocks.append(_Slot(tokens=list(tokens), key=key, parent_key=parent_key, k=k, v=v))

    def evict(self, index: int) -> KVBlock:
        """Забрать тензоры слота в KVBlock (слот становится заглушкой)."""
        slot = self.blocks[index]
        if not slot.resident:
            raise OverlayError(f"Slot {index} is already evicted.")
        block = KVBlock(tokens=list(slot.tokens), key=slot.key, k=slot.k, v=slot.v,
                        parent_key=slot.parent_key)
        slot.k = slot.v = None  # память освобождена
        return block

    def restore(self, index: int, block: KVBlock) -> None:
        """Вернуть тензоры в слот (точное совпадение токенов, ключа и родителя)."""
        slot = self.blocks[index]
        if slot.resident:
            raise OverlayError(f"Slot {index} is already resident.")
        if slot.tokens != list(block.tokens) or slot.key != block.key or slot.parent_key != block.parent_key:
            raise OverlayError("Restored block does not match slot identity.")
        slot.k, slot.v = block.k, block.v

    def total_tokens(self) -> int:
        return sum(len(s.tokens) for s in self.blocks)

    def resident_tokens(self) -> int:
        return sum(len(s.tokens) for s in self.blocks if s.resident)

    def evicted_slots(self) -> list[int]:
        return [i for i, s in enumerate(self.blocks) if not s.resident]

    def layer_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Полные K/V слоя по всем resident-блокам (вызов только при полном resident)."""
        if self.evicted_slots():
            raise OverlayError("KVCache has evicted slots: restore before attention.")
        return self._cat(layer, list(self.blocks))

    def _cat(self, layer: int, slots: list[_Slot]) -> tuple[torch.Tensor, torch.Tensor]:
        if not slots:
            empty = torch.zeros(1, self.spec.n_heads, 0, self.spec.head_dim)
            return empty, empty
        # slot.k[layer] = (H, len, hd) → [None] возвращает батч-ось → (1, H, T, hd)
        return (
            torch.cat([s.k[layer][None] for s in slots], dim=2),
            torch.cat([s.v[layer][None] for s in slots], dim=2),
        )


# ------------------------------------------------------- модель PoC


class KVTinyModel(nn.Module):
    """Крошечный причинный трансформер с ЯВНЫМ KV-кэшем (frozen, seeded).

    Слой: attention (n_heads) + residual; выход — логиты следующего токена.
    Нужен только чтобы цикл offload/restore проверялся бит-в-бит на настоящем
    attention вместо псевдо-LLM.
    """

    def __init__(self, vocab_size: int = 64, d: int = 32, n_heads: int = 2,
                 n_layers: int = 2, seed: int = 17) -> None:
        super().__init__()
        if d % n_heads:
            raise ValueError("d must be divisible by n_heads.")
        gen = torch.Generator().manual_seed(seed)
        self.vocab_size, self.d, self.n_heads, self.n_layers = vocab_size, d, n_heads, n_layers
        self.head_dim = d // n_heads
        self.embed = nn.Parameter(0.1 * torch.randn(vocab_size, d, generator=gen), requires_grad=False)
        self.wq = nn.ParameterList(
            nn.Parameter(0.05 * torch.randn(d, d, generator=gen), requires_grad=False) for _ in range(n_layers)
        )
        self.wk = nn.ParameterList(
            nn.Parameter(0.05 * torch.randn(d, d, generator=gen), requires_grad=False) for _ in range(n_layers)
        )
        self.wv = nn.ParameterList(
            nn.Parameter(0.05 * torch.randn(d, d, generator=gen), requires_grad=False) for _ in range(n_layers)
        )
        self.wo = nn.ParameterList(
            nn.Parameter(0.05 * torch.randn(d, d, generator=gen), requires_grad=False) for _ in range(n_layers)
        )
        self.w_out = nn.Parameter(0.05 * torch.randn(d, vocab_size, generator=gen), requires_grad=False)


# ------------------------------------------------------- хранилище блоков


def _kv_checksum(tokens: np.ndarray, k: np.ndarray, v: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(k.astype("<i1" if k.dtype == np.int8 else "<f4").tobytes())
    h.update(v.astype("<i1" if v.dtype == np.int8 else "<f4").tobytes())
    h.update(tokens.astype("<i8").tobytes())
    return h.hexdigest()


class KVBlockStore:
    """Таблица KV-блоков: точные ключи, LRU-слоты на устройстве, диск-тир.

    Совершенствованная часть: при выгрузке блок дистиллируется в D1-строки
    L2-субтаблицы (gist-ярус) — точное восстановление и «суть» живут в одной
    системе адресации и статистики.
    """

    def __init__(
        self,
        directory: str,
        spec: KVBlockSpec,
        model_tag: str = "kv0",
        n_slots: int = 4,
        device: torch.device | str | None = None,
        distill_spec: SubtableSpec | None = None,
        distill_scale: float = 0.05,
    ) -> None:
        self.directory = directory
        self.spec = spec
        self.model_tag = model_tag
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.n_slots = n_slots
        self.distill_spec = distill_spec
        self.distill_scale = float(distill_scale)
        self._distill_patch = None  # MemoryPatch для gist-яруса
        self._registry: dict[str, str] = {}            # key → путь npz («диск»)
        self._resident: OrderedDict[str, KVBlock] = OrderedDict()
        os.makedirs(directory, exist_ok=True)
        self.stats = {"puts": 0, "gets": 0, "hits": 0, "misses": 0, "evictions": 0, "total_load_ms": 0.0}

    # ---------------------------------------------------------- gist-ярус

    def _distill_write(self, block: KVBlock) -> None:
        """Биграмма-хвост блока → средний V блока (D1-блок-усреднение, как в L2)."""
        if self.distill_spec is None or len(block.tokens) < 2:
            return
        from ..memory import MemoryPatch
        from ..specs import TableSpec

        if self._distill_patch is None:
            table_spec = TableSpec(
                name=self.distill_spec.name, n_rows=self.distill_spec.n_rows,
                row_width=self.distill_spec.row_width, n_heads=1,
                key_order=self.distill_spec.key_order, inject_layer=2,
            )
            self._distill_patch = MemoryPatch(table_spec)
        # адрес — канонический blake2b биграммы (совместим с L1/L2 адресацией)
        from .hashing import blake_addresses

        row = blake_addresses(list(block.tokens), self.distill_spec.n_rows, 2)[-1]
        vec = block.v.mean(dim=(0, 2)).flatten()  # (H*hd,) — gist V: среднее по слоям и позициям
        vec = vec / max(float(vec.norm()), 1e-8)
        target = (self.distill_scale * vec).cpu().numpy().astype(np.float32)
        D1Rule().apply(self._distill_patch, [Event(row_id=int(row), target=target, gate=0.5)])

    def distill_table(self) -> L2Subtable | None:
        """Материализовать gist-ярус в плотную L2-субтаблицу (None без дистилляции)."""
        if self._distill_patch is None:
            return None
        return L2Subtable.from_patch(self._distill_patch, self.distill_spec)

    def gist(self, tokens: list[int]) -> torch.Tensor | None:
        """«Суть» контекста без точного восстановления: строка gist-яруса или None."""
        table = self.distill_table()
        if table is None or len(tokens) < 2:
            return None
        from .hashing import blake_addresses

        row = torch.tensor([[blake_addresses(list(tokens), self.distill_spec.n_rows, 2)[-1]]])
        vals, hits = table.lookup(row)
        return vals[0, 0] if bool(hits[0, 0]) else None

    # ------------------------------------------------------------ диск

    def _path(self, key: str) -> str:
        return os.path.join(self.directory, f"{key[:40]}.kvoff.npz")

    def _save(self, block: KVBlock) -> None:
        path = self._path(block.key)
        tokens = np.asarray(block.tokens, dtype=np.int64)
        k = block.k.detach().cpu().numpy()
        v = block.v.detach().cpu().numpy()
        meta = {"spec": asdict(self.spec), "model_tag": self.model_tag,
                "key": block.key, "parent_key": block.parent_key,
                "kvoff_format": KVOFF_FORMAT, "kvoff_version": KVOFF_VERSION}
        if self.spec.storage == "int8":
            ks = max(float(np.abs(k).max()) / 127.0, 1e-12)
            vs = max(float(np.abs(v).max()) / 127.0, 1e-12)
            meta["integrity"] = _kv_checksum(tokens, np.rint(k / ks).astype(np.int8), np.rint(v / vs).astype(np.int8))
            np.savez(path, tokens=tokens, k_i8=np.rint(k / ks).astype(np.int8),
                     v_i8=np.rint(v / vs).astype(np.int8),
                     k_scale=np.float32(ks), v_scale=np.float32(vs), meta=np.asarray(json.dumps(meta)))
        else:
            meta["integrity"] = _kv_checksum(tokens, k.astype(np.float32), v.astype(np.float32))
            np.savez(path, tokens=tokens, k=k.astype(np.float32), v=v.astype(np.float32),
                     meta=np.asarray(json.dumps(meta)))

    def _load(self, key: str) -> KVBlock:
        with np.load(self._registry[key], allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("kvoff_format") != KVOFF_FORMAT or meta.get("kvoff_version") != KVOFF_VERSION:
                raise OverlayError(f"kvoff block {key[:12]!r}: unsupported format/version")
            tokens = [int(t) for t in data["tokens"]]
            parent_key = meta.get("parent_key")
            if meta.get("key") != key or block_key(self.model_tag, tokens, parent_key) != key:
                raise OverlayError(f"kvoff block {key[:12]!r}: key/tokens mismatch")
            if self.spec.storage == "int8":
                k = torch.as_tensor(np.asarray(data["k_i8"]), dtype=torch.float32) * float(data["k_scale"])
                v = torch.as_tensor(np.asarray(data["v_i8"]), dtype=torch.float32) * float(data["v_scale"])
            else:
                k = torch.as_tensor(np.asarray(data["k"]), dtype=torch.float32)
                v = torch.as_tensor(np.asarray(data["v"]), dtype=torch.float32)
            expected = meta.get("integrity")
            if expected is not None:
                tk = np.asarray(data["tokens"], dtype=np.int64)
                if self.spec.storage == "int8":
                    actual = _kv_checksum(tk, np.asarray(data["k_i8"]), np.asarray(data["v_i8"]))
                else:
                    actual = _kv_checksum(tk, np.asarray(data["k"]), np.asarray(data["v"]))
                if actual != expected:
                    raise OverlayError(f"kvoff block {key[:12]!r}: integrity check failed")
        return KVBlock(tokens=tokens, key=key, k=k.to(self.device), v=v.to(self.device),
                       parent_key=parent_key)

    # ------------------------------------------------------- LRU-доступ

    def put(self, block: KVBlock) -> None:
        """Записать блок на «диск» (+ gist-дистилляция), внести в реестр."""
        if block_key(self.model_tag, block.tokens, block.parent_key) != block.key:
            raise OverlayError("KVBlock key does not match its tokens/parent chain.")
        self._save(block)
        self._registry[block.key] = self._path(block.key)
        self._distill_write(block)
        self.stats["puts"] += 1

    def has(self, key: str) -> bool:
        return key in self._registry

    def get(self, key: str) -> KVBlock | None:
        """Блок на устройстве (LRU). Промах — None (решение о recompute — у сессии)."""
        self.stats["gets"] += 1
        if key in self._resident:
            self._resident.move_to_end(key)
            self.stats["hits"] += 1
            return self._resident[key]
        if key not in self._registry:
            self.stats["misses"] += 1
            return None
        while len(self._resident) >= self.n_slots:
            self._resident.popitem(last=False)
            self.stats["evictions"] += 1
        t0 = time.perf_counter()
        block = self._load(key)
        self.stats["total_load_ms"] += (time.perf_counter() - t0) * 1000.0
        self._resident[key] = block
        self.stats["hits"] += 1
        return block

    def stats_summary(self) -> dict:
        total = self.stats["hits"] + self.stats["misses"]
        return {
            "puts": self.stats["puts"],
            "gets": self.stats["gets"],
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": round(self.stats["hits"] / total, 3) if total else 0.0,
            "evictions": self.stats["evictions"],
            "mean_load_ms": round(self.stats["total_load_ms"] / max(self.stats["hits"], 1), 3),
            "n_blocks_on_disk": len(self._registry),
        }


# ------------------------------------------------------------- политика


@dataclass
class OffloadPolicy:
    """sink + свежее окно остаются resident, середина уходит в таблицу."""

    sink_blocks: int = 1
    recent_blocks: int = 1

    def targets(self, cache: KVCache) -> list[int]:
        n = len(cache.blocks)
        lo, hi = self.sink_blocks, n - self.recent_blocks
        return list(range(lo, max(hi, lo))) if hi > lo else []


# -------------------------------------------------------------- сессия


@dataclass
class PrefillStats:
    """Учёт prefill: сколько блоков восстановлено из таблицы, пересчитано, записано."""

    restored: int = 0
    computed: int = 0
    stored: int = 0


class KVOffloadSession:
    """Генерация с явным KV-кэшем: prefill (с переиспользованием таблицы) →
    offload (освобождение памяти) → forward с restore-before-forward."""

    def __init__(self, model: KVTinyModel, store: KVBlockStore | None = None,
                 block_len: int = 4, device: torch.device | str | None = None) -> None:
        self.model = model.to(device) if device is not None else model
        self.device = next(self.model.parameters()).device
        self.store = store
        if store is not None and (store.spec.n_layers != model.n_layers
                                  or store.spec.n_heads != model.n_heads
                                  or store.spec.head_dim != model.head_dim):
            raise OverlayError("KVBlockStore spec does not match the model.")
        self.block_len = store.spec.block_len if store is not None else block_len
        self.model_tag = store.model_tag if store is not None else "kv0"
        self._spec = KVBlockSpec(
            n_layers=model.n_layers, n_heads=model.n_heads, head_dim=model.head_dim,
            block_len=self.block_len,
            storage=store.spec.storage if store is not None else "fp32",
        )

    def new_cache(self) -> KVCache:
        return KVCache(self._spec)

    # --- ядро вычисления одного куска поверх resident-префикса кэша

    def _forward_chunk(self, tokens: list[int], cache: KVCache, *, reuse_kv: bool = False):
        """Внимание нового куска поверх resident-префикса; возвращает логиты и KV куска.

        reuse_kv=True — logits-only проход: KV этих позиций УЖЕ в кэше (полностью
        восстановленный prefill); логиты считаются без повторного добавления KV.
        """
        m = self.model
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        x = m.embed[ids]                                  # (1, L, d)
        start = cache.total_tokens() - (len(tokens) if reuse_kv else 0)
        chunk_k, chunk_v = [], []
        for li in range(m.n_layers):
            q = (x @ m.wq[li]).view(1, -1, m.n_heads, m.head_dim).transpose(1, 2)
            k = (x @ m.wk[li]).view(1, -1, m.n_heads, m.head_dim).transpose(1, 2)
            v = (x @ m.wv[li]).view(1, -1, m.n_heads, m.head_dim).transpose(1, 2)
            if reuse_kv:
                K, V = cache.layer_kv(li)                  # включает KV самих позиций
            else:
                past_k, past_v = cache.layer_kv(li)
                K = torch.cat([past_k, k], dim=2)
                V = torch.cat([past_v, v], dim=2)
            qi = torch.arange(start, start + q.shape[2], device=self.device)
            kj = torch.arange(K.shape[2], device=self.device)
            mask = kj[None, :] <= qi[:, None]             # каузальность внутри куска
            scores = (q @ K.transpose(-1, -2)) / (m.head_dim ** 0.5)
            attn = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
            ctx = (attn @ V).transpose(1, 2).reshape(1, -1, m.d)
            x = x + ctx @ m.wo[li]                         # residual
            chunk_k.append(k)
            chunk_v.append(v)
        logits = x @ m.w_out                              # (1, L, V)
        # (n_layers, H, L, hd): слой — первая ось, батч-единица сжимается
        return logits, torch.stack(chunk_k, dim=0)[:, 0], torch.stack(chunk_v, dim=0)[:, 0]

    # --- точное восстановление с fallback'ом на пересчёт

    def _ensure_resident(self, cache: KVCache) -> None:
        """Restore-before-forward: все слоты resident; точный промах → recompute."""
        for i, slot in enumerate(cache.blocks):
            if slot.resident:
                continue
            if self.store is None:
                raise OverlayError("Evicted slot without a store cannot be restored.")
            block = self.store.get(slot.key)
            if block is None:
                # recompute-fallback: честный пересчёт куска поверх resident-префикса
                logits, k, v = self._forward_chunk(slot.tokens, _ResidentPrefix(cache, i))
                block = KVBlock(tokens=list(slot.tokens), key=slot.key, k=k, v=v,
                                parent_key=slot.parent_key)
            cache.restore(i, block)

    def prefill(self, tokens: list[int], cache: KVCache, *, cache_to_store: bool = True):
        """Кусками по block_len: хит таблицы → restore без вычислений; иначе compute.

        Возвращает (логиты последнего куска, PrefillStats). KV куска зависит от
        всего префикса, поэтому ключ блока — hash(префикс кэша + кусок).
        """
        stats = PrefillStats()
        logits = None
        for w in range(0, len(tokens), self.block_len):
            chunk = [int(t) for t in tokens[w : w + self.block_len]]
            parent = cache.blocks[-1].key if cache.blocks else None
            key = block_key(self.model_tag, chunk, parent)
            block = self.store.get(key) if self.store is not None else None
            if block is not None:
                cache.append_block(block.tokens, block.key, block.k, block.v, block.parent_key)
                stats.restored += 1
            else:
                logits, k, v = self._forward_chunk(chunk, cache)
                cache.append_block(chunk, key, k, v, parent)
                stats.computed += 1
                if self.store is not None and cache_to_store:
                    self.store.put(KVBlock(tokens=chunk, key=key, k=k, v=v, parent_key=parent))
                    stats.stored += 1
        if logits is None:
            # весь префикс восстановлен из таблицы — логиты logits-only проходом
            logits, _, _ = self._forward_chunk(list(cache.blocks[-1].tokens), cache, reuse_kv=True)
        return logits, stats

    def offload(self, cache: KVCache, policy: OffloadPolicy | None = None) -> dict:
        """Выгрузить середину кэша в таблицу: память освобождается, токены остаются."""
        if self.store is None:
            raise OverlayError("offload requires a KVBlockStore.")
        policy = policy or OffloadPolicy()
        targets = policy.targets(cache)
        freed = 0
        for i in targets:
            block = cache.evict(i)
            if not self.store.has(block.key):
                self.store.put(block)
            freed += block.length
        return {
            "offloaded_blocks": len(targets),
            "freed_tokens": freed,
            "resident_tokens": cache.resident_tokens(),
            "total_tokens": cache.total_tokens(),
        }

    def forward(self, tokens: list[int], cache: KVCache):
        """Один шаг генерации: restore-before-forward → attention → новый блок."""
        self._ensure_resident(cache)
        toks = [int(t) for t in tokens]
        parent = cache.blocks[-1].key if cache.blocks else None
        logits, k, v = self._forward_chunk(toks, cache)
        cache.append_block(toks, block_key(self.model_tag, toks, parent), k, v, parent)
        return logits

    def _prefix_tokens(self, cache: KVCache) -> list[int]:
        """Все токены resident-куска префикса (слоты кэша — источник истины)."""
        return [t for slot in cache.blocks for t in slot.tokens]


class _ResidentPrefix:
    """View кэша «только слоты до i» — для recompute-пересчёта вытесненного блока."""

    def __init__(self, cache: KVCache, upto: int) -> None:
        self._cache = cache
        self._upto = upto
        self.spec = cache.spec

    def total_tokens(self) -> int:
        return sum(len(s.tokens) for s in self._cache.blocks[: self._upto])

    def layer_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        slots = list(self._cache.blocks[: self._upto])
        if any(not s.resident for s in slots):
            raise OverlayError("Recompute prefix must be fully resident.")
        return self._cache._cat(layer, slots)


def greedy_generate(session: KVOffloadSession, cache: KVCache, prompt: list[int], n_steps: int) -> list[int]:
    """Жадная генерация n_steps токенов через сессию (prefill → forward с restore)."""
    logits, _ = session.prefill(list(prompt), cache)
    out = [int(logits[0, -1].argmax())]
    for _ in range(n_steps - 1):
        logits = session.forward([out[-1]], cache)
        out.append(int(logits[0, -1].argmax()))
    return out
