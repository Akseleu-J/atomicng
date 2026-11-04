"""conftest: снятие пиков памяти между тестами (TinyModel ≈ 10.5 МБ на инстанс)."""
from __future__ import annotations

import gc

import pytest


@pytest.fixture(autouse=True)
def _gc_between_tests():
    yield
    gc.collect()
