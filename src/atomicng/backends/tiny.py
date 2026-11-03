"""Детерминированная CPU-модель для тестов и CI (аналог smoke-конфига официального репо).

НЕ модель статьи: крошечный псевдо-LLM, который читает таблицу T⁰+Δ так же, как нативный
движок (gather строк по хэшу биграммы), — этого достаточно, чтобы гонять весь цикл
write → read → clear → verify на любом CPU без torch.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..memory import MemoryPatch
from ..specs import SIDECAR_SMALL, TableSpec


def _logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


class TinyModel:
    """Псевдо-LLM: эмбеддинги токенов + чтение n-gram таблицы перед «выходным слоем».

    Логиты для последовательности токенов: пул эмбеддингов + пул прочитанных строк
    таблицы, спроецированные фиксированной матрицей. Всё детерминировано (blake2b-хэш
    строк таблицы, seeded RNG весов).
    """

    def __init__(
        self,
        spec: TableSpec | None = None,
        vocab_size: int = 512,
        out_dim: int = 32,
        seed: int = 17,
    ) -> None:
        self.spec = spec or SIDECAR_SMALL
        self.vocab_size = vocab_size
        rng = np.random.default_rng(seed)
        self.t0 = (0.02 * rng.standard_normal((self.spec.n_rows, self.spec.row_width))).astype(np.float32)
        self.embed = (0.10 * rng.standard_normal((vocab_size, self.spec.row_width))).astype(np.float32)
        self.w_out = (0.05 * rng.standard_normal((self.spec.row_width, out_dim))).astype(np.float32)
        self.w_vocab = (0.05 * rng.standard_normal((self.spec.row_width, vocab_size))).astype(np.float32)
        self.memory = MemoryPatch(self.spec)

    @staticmethod
    def row_hash(prev_token: int, token: int, n_rows: int) -> int:
        """Детерминированный кросс-платформенный хэш биграммы → номер строки."""
        digest = hashlib.blake2b(f"{int(prev_token)}:{int(token)}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "little") % n_rows

    def read(self, tokens: list[int]) -> np.ndarray:
        """Чтение таблицы по биграммам: (len, d), позиция 0 — нули (нет предшественника)."""
        out = np.zeros((len(tokens), self.spec.row_width), dtype=np.float32)
        for i in range(1, len(tokens)):
            row = self.row_hash(tokens[i - 1], tokens[i], self.spec.n_rows)
            out[i] = self.t0[row] + self.memory.get(row)
        return out

    def logits(self, tokens: list[int]) -> np.ndarray:
        """(out_dim,) — «выход модели» для последовательности."""
        h = self.embed[tokens].mean(axis=0) + self.read(tokens).mean(axis=0)
        return h @ self.w_out

    def probe(self, n: int = 8, seq_len: int = 12, seed: int = 7) -> np.ndarray:
        """Детерминированный набор выходов: (n, out_dim). Не гарантирует нейтральности."""
        rng = np.random.default_rng(seed)
        tokens = rng.integers(0, self.vocab_size, size=(n, seq_len))
        return np.stack([self.logits([int(t) for t in seq]) for seq in tokens])

    def neutral_token_seqs(self, n: int = 8, seq_len: int = 12, seed: int = 7, max_tries: int = 500) -> list[list[int]]:
        """Последовательности, гарантированно НЕ адресующие строки текущего патча.

        Генерируются один раз и переиспользуются до и после очистки памяти — так
        verify/publish сравнивают выходы на одних и тех же входах.
        """
        written = set(self.memory.row_ids()) if self.memory is not None else set()
        rng = np.random.default_rng(seed)
        seqs: list[list[int]] = []
        tries = 0
        while len(seqs) < n and tries < max_tries:
            tries += 1
            tokens = [int(x) for x in rng.integers(0, self.vocab_size, size=seq_len)]
            rows = {self.row_hash(tokens[i - 1], tokens[i], self.spec.n_rows) for i in range(1, seq_len)}
            if rows & written:
                continue
            seqs.append(tokens)
        if len(seqs) < n:
            raise RuntimeError(f"не удалось набрать {n} нейтральных последовательностей (патч слишком плотный)")
        return seqs

    def run_seqs(self, seqs: list[list[int]]) -> np.ndarray:
        return np.stack([self.logits(seq) for seq in seqs])

    def token_logits(self, tokens: list[int]) -> np.ndarray:
        """Логиты следующего токена (словарные, для NLL): (T-1, V)."""
        h = self.embed[tokens[:-1]] + self.read(tokens[:-1])
        return h @ self.w_vocab

    def nll(self, seqs: list[list[int]]) -> float:
        """Средний NLL (nats/token) предсказания t+1 при текущей памяти T⁰+Δ."""
        total, count = 0.0, 0
        for seq in seqs:
            if len(seq) < 2:
                continue
            logits = self.token_logits(seq)
            logprobs = logits - _logsumexp(logits, axis=-1, keepdims=True)
            targets = np.asarray(seq[1:], dtype=np.int64)
            total -= float(np.sum(logprobs[np.arange(len(targets)), targets]))
            count += len(targets)
        return total / max(count, 1)
