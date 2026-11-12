"""conftest: Removing memory peaks between tests (TinyModel ≈ 10.5 MB per instance))."""
from __future__ import annotations

import gc

import pytest


@pytest.fixture(autouse=True)
def _gc_between_tests():
    yield
    gc.collect()
