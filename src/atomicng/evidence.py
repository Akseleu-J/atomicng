"""Конструктор свидетельств (T10а — то, чего нет ни у одного конкурента).

Документ (последовательность токенов) → блоки «query-префикс 128 + продолжение до 512»
по протоколу обучения/записи статьи. Токенизатор — на выбор пользователя; для беззависимостных
сценариев есть ByteTokenizer (UTF-8 байты → id, детерминированный).
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    """Блок свидетельства: query — префикс-контекст, continuation — часть для записи/предсказания."""

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
    """Режет токены на блоки статьи: 512 токенов, из них 128 — query-префикс.

    stride по умолчанию = block_size − query_prefix (непересекающиеся продолжения).
    """
    if block_size <= query_prefix:
        raise ValueError(f"block_size={block_size} должен быть больше query_prefix={query_prefix}")
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
    """Удобство: строки → байты-токены → блоки (для tiny-пайплайна и тестов)."""
    tokenizer = tokenizer or ByteTokenizer()
    blocks: list[Block] = []
    for i, text in enumerate(texts):
        blocks.extend(build_blocks(tokenizer.encode(text), source=f"{source_prefix}-{i}", **kwargs))
    return blocks


class ByteTokenizer:
    """Детерминированный токенизатор без зависимостей: UTF-8 байты, свёрнутые в словарь."""

    def __init__(self, vocab_size: int = 512) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size ByteTokenizer должен быть >= 256 (байты)")
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
