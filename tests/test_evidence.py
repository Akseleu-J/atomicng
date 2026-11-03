"""Тесты evidence-пайплайна: блоки 512/128 по протоколу статьи, токенизатор."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomicng import ByteTokenizer, build_blocks, build_blocks_from_texts


def test_block_geometry():
    tokens = list(range(1000))
    blocks = build_blocks(tokens)
    assert all(len(b.query) == 128 for b in blocks)
    assert all(len(b.query) + len(b.continuation) <= 512 for b in blocks)
    assert blocks[0].tokens == tokens[:512]
    # stride по умолчанию 384 → начала 0, 384, 768
    assert blocks[1].tokens[0] == 384 and blocks[2].tokens[0] == 768
    assert [b.index for b in blocks] == [0, 1, 2]
    assert all(b.source == "doc" for b in blocks)


def test_short_document_single_query_only_rejected():
    assert build_blocks([]) == []
    assert build_blocks(list(range(50))) == []  # короче префикса — блоков нет


def test_invalid_geometry():
    with pytest.raises(ValueError):
        build_blocks([1], block_size=128, query_prefix=128)


def test_byte_tokenizer_deterministic():
    tok = ByteTokenizer(vocab_size=512)
    a = tok.encode("кенгуру")
    assert a == tok.encode("кенгуру")
    assert all(0 <= i < 512 for i in a)
    with pytest.raises(ValueError):
        ByteTokenizer(vocab_size=128)


def test_build_from_texts_sources():
    blocks = build_blocks_from_texts(["a" * 600, "b" * 200])
    assert blocks[0].source == "doc-0" and blocks[-1].source == "doc-1"
    assert all(len(b.query) == 128 for b in blocks)
