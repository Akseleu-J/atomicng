"""Reference constants from arXiv:2610.31125.

Conformance tests check the implementation against these values.
See FIDELITY.md for documented deviations.
"""
from __future__ import annotations

# --- table ----------------------------------------------------------------
NATIVE_ROW_WIDTH = 160          # d: Native table row width Qwen3.8-Flash-Next
NATIVE_HASH_HEADS = 16          # H: 8 bigrams + 8 trigram heads
NATIVE_HEADS_BIGRAM = 8
NATIVE_HEADS_TRIGRAM = 8
KEY_ORDERS = (2, 3)             # keys - hashes of the last 2-3 tokens
INJECT_LAYER = 2                # Reading before attention 2nd layer

# --- writer -----------------------------------------------------------------
WRITER_HIDDEN = 256             # Two 2-layer MLPs, hidden 256
WRITER_PARAM_COUNT = 4_075_681  # The parameters of the native writer article
TARGET_RMS_RATIO = 0.01         # ρ: RMS of initial targets = ρ·RMS(T)⁰_h) (fixed-s_h)
FEATURE_NEXT_WINDOW = 4         # x_a: average of the following 4 positions (Eq. 7))
GATE_INPUT_LOG_COUNT = True     # β_a = σ(B_θ[x_a, Δ_j, v_a−Δ_j, log(1+c_j)])

# --- rule D1 ------------------------------------------------------
DECAY_ALPHA_DEFAULT = 1.0       # Δ_j ← α·Δ_j; main config of the article - without fading
DECAY_ALPHA_COMPARED = 0.99     # comparison

# --- writer-training ------------------------------------------------------
BLOCK_SIZE = 512                # 512 token blocks
QUERY_PREFIX = 128              # query prefix 128 tokens
GROUP_TRAJECTORIES = 12         # Loss is normalized once per group of 12 trajectories
LEARNING_RATE = 3e-4            # AdamW
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
TRAIN_TOKENS = 100_000_000      # budget α=1
TRAIN_TOKENS_ALPHA099 = 40_000_000
DELTA_FORK_TOKENS = 80_000_000  # Delta-writer forks with plain and learns +20M

# --- Results of the article (reference numbers for future reproductions)) --------
DEV64_DELTA_EMPTY = 0.068       # nats/token, plain writer
DEV64_DELTA_DONOR = 0.024
RETENTION_K16_PLAIN = 0.54
RETENTION_K16_DELTA = 0.70
CHAIN_WRITES = 35_000
HOT_ROW_NORM_RATIO = (4.4, 11.7)  # Formatting-row norm inflation relative to ‖T⁰_j‖

__all__ = [name for name in dir() if name.isupper()]
