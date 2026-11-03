"""Канонические константы статьи (arXiv:2610.31125) — единственный источник истины.

Каждая константа имеет проверяемый тест (tests/test_fidelity.py): реализация не имеет права
тихо отклониться. Отклонения допустимы только явные (mini-конфиги CPU, sidecar-адаптер) и
задокументированы в FIDELITY.md.
"""
from __future__ import annotations

# --- таблица ----------------------------------------------------------------
NATIVE_ROW_WIDTH = 160          # d: ширина строки нативной таблицы Qwen3.8-Flash-Next
NATIVE_HASH_HEADS = 16          # H: 8 биграммных + 8 триграммных голов
NATIVE_HEADS_BIGRAM = 8
NATIVE_HEADS_TRIGRAM = 8
KEY_ORDERS = (2, 3)             # ключи — хэши последних 2–3 токенов
INJECT_LAYER = 2                # чтение перед attention 2-го слоя

# --- writer -----------------------------------------------------------------
WRITER_HIDDEN = 256             # два 2-слойных MLP, hidden 256
WRITER_PARAM_COUNT = 4_075_681  # параметры нативного writer'а статьи
TARGET_RMS_RATIO = 0.01         # ρ: RMS начальных целей = ρ·RMS(T⁰_h) (фиксированные s_h)
FEATURE_NEXT_WINDOW = 4         # x_a: среднее по следующим 4 позициям (Eq. 7)
GATE_INPUT_LOG_COUNT = True     # β_a = σ(B_θ[x_a, Δ_j, v_a−Δ_j, log(1+c_j)])

# --- правило записи D1 ------------------------------------------------------
DECAY_ALPHA_DEFAULT = 1.0       # Δ_j ← α·Δ_j; основной конфиг статьи — без затухания
DECAY_ALPHA_COMPARED = 0.99     # сравнивавшийся в статье вариант

# --- обучение writer'а ------------------------------------------------------
BLOCK_SIZE = 512                # блоки 512 токенов
QUERY_PREFIX = 128              # query-префикс 128 токенов
GROUP_TRAJECTORIES = 12         # loss нормируется раз на группу из 12 траекторий
LEARNING_RATE = 3e-4            # AdamW
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
TRAIN_TOKENS = 100_000_000      # бюджет α=1
TRAIN_TOKENS_ALPHA099 = 40_000_000
DELTA_FORK_TOKENS = 80_000_000  # delta-writer форкуется с plain и доучивается +20M

# --- результаты статьи (эталонные числа для будущих воспроизведений) --------
DEV64_DELTA_EMPTY = 0.068       # nats/token, plain writer
DEV64_DELTA_DONOR = 0.024
RETENTION_K16_PLAIN = 0.54
RETENTION_K16_DELTA = 0.70
CHAIN_WRITES = 35_000
HOT_ROW_NORM_RATIO = (4.4, 11.7)  # раздутие форматных строк относительно ‖T⁰_j‖

__all__ = [name for name in dir() if name.isupper()]
