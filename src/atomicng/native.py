"""Track B n-gram hashing and native read-out.

The paper uses eight bigram and eight trigram heads of width 160. Concatenated
reads form a 2,560-dimensional vector. The value and gated key paths produce a
10,240-dimensional residual input. See FIDELITY.md for the documented
interpretation of the gated key path.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn

from .paper import (
    NATIVE_HEADS_BIGRAM,
    NATIVE_HEADS_TRIGRAM,
    NATIVE_ROW_WIDTH,
)


class NativeTableSpec:
    """Specification of the factory table: 16 heads (8+8), d=160, keys 2-3 tokens."""

    def __init__(
        self,
        name: str = "native-qwen38",
        n_rows: int = 16384,
        row_width: int = NATIVE_ROW_WIDTH,
        bigram_heads: int = NATIVE_HEADS_BIGRAM,
        trigram_heads: int = NATIVE_HEADS_TRIGRAM,
        inject_layer: int = 2,
    ) -> None:
        if row_width != NATIVE_ROW_WIDTH:
            raise ValueError(f"The paper fixes the native row width at {NATIVE_ROW_WIDTH}")
        self.name = name
        self.n_rows = n_rows
        self.row_width = row_width
        self.bigram_heads = bigram_heads
        self.trigram_heads = trigram_heads
        self.n_heads = bigram_heads + trigram_heads
        self.inject_layer = inject_layer

    @property
    def concat_dim(self) -> int:
        return self.n_heads * self.row_width  # 2560


def native_head_rows(
    input_ids: list[int], spec: NativeTableSpec, head: int
) -> list[int]:
    """Addresses of one head rows for each position (i ≥ order-1); None - position without a key.

    Heads 0..bigram-1 hash bigrams (t−1, t), heads bigram.. - trigrams (t−2, t−1, t)).
    Head salt eliminates the coincidence of addresses between the heads.
    """
    order = 2 if head < spec.bigram_heads else 3
    out: list[int | None] = [None] * len(input_ids)
    for i in range(len(input_ids)):
        if i < order - 1:
            continue
        window = input_ids[i - order + 1 : i + 1]
        digest = hashlib.blake2b(
            f"{head}:{','.join(map(str, window))}".encode(), digest_size=8
        ).digest()
        out[i] = int.from_bytes(digest, "little") % spec.n_rows
    return out


def native_rows_all_heads(input_ids: list[int], spec: NativeTableSpec) -> list[list[int | None]]:
    """Return row addresses as an (H, T) array for all 16 heads of T⁰."""
    return [native_head_rows(input_ids, spec, h) for h in range(spec.n_heads)]


class NativeReadOut(nn.Module):
    """Read-out: concat(H·d) → value-path + Gated key path - R^out_dim (10240)."""

    def __init__(
        self,
        spec: NativeTableSpec,
        out_dim: int = 10240,
        conv_taps: int = 4,
        dilation: int = 3,
        seed: int = 17,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.out_dim = out_dim
        gen = torch.Generator().manual_seed(seed)
        self.t0 = 0.02 * torch.randn(spec.n_rows, spec.row_width, generator=gen)  # T⁰ (immutable)
        in_dim = spec.concat_dim
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.RMSNorm(out_dim)
        self.conv = nn.Conv1d(
            out_dim, out_dim, kernel_size=conv_taps, dilation=dilation,
            groups=out_dim, bias=False,  # depthwise
        )
        nn.init.normal_(self.value.weight, std=0.01, generator=gen)
        nn.init.normal_(self.key.weight, std=0.01, generator=gen)
        nn.init.normal_(self.conv.weight, std=0.01, generator=gen)

    def gather(self, input_ids: list[int], memory_row_fn) -> torch.Tensor:
        """Return (T, H·d) rows from T⁰+Δ across all heads. Positions with no key are zero. memory_row_fn(row)->Tensor|None."""
        rows = native_rows_all_heads(input_ids, self.spec)
        t = len(input_ids)
        out = torch.zeros(t, self.spec.n_heads, self.spec.row_width)
        for h, head_rows in enumerate(rows):
            for i, r in enumerate(head_rows):
                if r is None:
                    continue
                piece = self.t0[r]
                extra = memory_row_fn(r)
                if extra is not None:
                    piece = piece + extra
                out[i, h] = piece
        return out.reshape(t, self.spec.concat_dim)

    @staticmethod
    def sign_sqrt_max_floor(k: torch.Tensor) -> torch.Tensor:
        """sign/ √max floor: compression of dynamic range without loss of sign."""
        denom = k.abs().amax(dim=-1, keepdim=True) + 1e-6
        return torch.sign(k) * torch.sqrt(k.abs() / denom)

    def forward(self, concat_rows: torch.Tensor) -> torch.Tensor:
        """(B, T, H·d) → (B, T, out_dim). Causalities: Conv looks only into the past (dilation, padding left))."""
        b, t, _ = concat_rows.shape
        v = self.value(concat_rows)
        k = self.norm(self.key(concat_rows))
        k = self.sign_sqrt_max_floor(k)
        pad = (self.conv.dilation[0]) * (self.conv.kernel_size[0] - 1)  # causal
        k = k.transpose(1, 2)
        k = torch.nn.functional.pad(k, (pad, 0))
        g = self.conv(k).transpose(1, 2)
        return v * torch.sigmoid(g)


__all__ = ["NativeTableSpec", "NativeReadOut", "native_head_rows", "native_rows_all_heads"]
