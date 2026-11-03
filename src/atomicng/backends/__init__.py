"""Engine-адаптеры (слой чтения таблицы моделью).

v0.1: tiny — детерминированная CPU-модель для CI.
v0.2: torch/HF attach. v0.4: vLLM-ридер, llama.cpp PLE-экспорт.
"""

from .tiny import TinyModel

__all__ = ["TinyModel"]
