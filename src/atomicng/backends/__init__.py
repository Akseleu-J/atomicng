"""Backend adapters for n-gram table reads.

TinyModel is a deterministic CPU backend for tests. The optional PyTorch
adapter attaches a sidecar to a Hugging Face causal language model.
"""

from .tiny import TinyModel

__all__ = ["TinyModel"]
