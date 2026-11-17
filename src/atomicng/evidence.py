"""Build evidence blocks for training and writes.

The paper uses blocks of up to 512 tokens with a 128-token query prefix.
ByteTokenizer provides deterministic UTF-8 byte IDs for standalone examples.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    """An evidence block contains a query prefix and a continuation for writing or prediction."""

    query: list[int]
    continuation: list[int]
    source: str
    index: int

    @property
    def tokens(self) -> list[int]:
        return self.query + self.continuation


def build_blocks(
    tokens: Sequence[int],
    *,
    source: str = "doc",
    block_size: int = 512,
    query_prefix: int = 128,
    stride: int | None = None,
) -> list[Block]:
    """Cuts tokens into blocks of article: 512 tokens, of which 128 are query prefix.

    stride by default = block_size − query_prefix (intersecting).
    """
    if block_size <= query_prefix:
        raise ValueError(f"block_size={block_size} must exceed query_prefix={query_prefix}.")
    if not tokens:
        return []
    stride = stride or (block_size - query_prefix)
    blocks: list[Block] = []
    start = 0
    index = 0
    while start < len(tokens):
        chunk = list(tokens[start : start + block_size])
        if len(chunk) > query_prefix:
            blocks.append(
                Block(
                    query=chunk[:query_prefix],
                    continuation=chunk[query_prefix:block_size],
                    source=source,
                    index=index,
                )
            )
            index += 1
        if len(chunk) < block_size:
            break
        start += stride
    return blocks


def build_blocks_from_texts(
    texts: Sequence[str],
    *,
    tokenizer: ByteTokenizer | None = None,
    source_prefix: str = "doc",
    **kwargs,
) -> list[Block]:
    """Convenience: strings → byte-tokens → blocks (for tiny-pipeline and tests))."""
    tokenizer = tokenizer or ByteTokenizer()
    blocks: list[Block] = []
    for i, text in enumerate(texts):
        blocks.extend(build_blocks(tokenizer.encode(text), source=f"{source_prefix}-{i}", **kwargs))
    return blocks


class ByteTokenizer:
    """Deterministic tokenizer without dependencies: UTF-8 bytes folded into a dictionary."""

    def __init__(self, vocab_size: int = 512) -> None:
        if vocab_size < 256:
            raise ValueError("ByteTokenizer requires a vocabulary of at least 256 byte values.")
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [b % self.vocab_size for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        inv = {b % self.vocab_size: b for b in range(self.vocab_size)}
        return bytes(inv.get(i, 32) for i in ids).decode("utf-8", errors="replace")

    def iter_chunks(self, text: str, size: int) -> Iterator[list[int]]:
        ids = self.encode(text)
        for i in range(0, len(ids), size):
            yield ids[i : i + size]
