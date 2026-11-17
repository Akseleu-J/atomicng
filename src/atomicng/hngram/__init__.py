"""H-Ngram: иерархическая внешняя память L2 поверх ядра AtomicNG.

База одной L2-субтаблицы. Компоненты:

- hashing  — адресация n-грамм: blake2b (совместим с NumPy-ядром) и poly64
  (векторизованный GPU-путь);
- store    — L2Subtable: плотная квантованная таблица, собранная из MaterialPatch
  (наполнение через тот же D1), integrity-манифест как у overlay;
- router   — TopicRouter: лёгкий классификатор темы с гистерезисом (без thrash);
- manager  — SubtableManager: реестр субтаблиц + LRU-слоты на устройстве,
  горячая подмена «тома», статистика переключений;
- fill     — детерминированный наполнитель субтаблицы (HashWriter-стиль):
  строка биграммы хранит продолжение (delta-смысл строки);
- sidecar  — L2Sidecar: инъекция γ·Δ_L2 в резидуальный поток TorchTinyModel,
  γ zero-init → при attach выходы бит-в-бит равны базовым.

Требует опциональный слой PyTorch: pip install 'atomicng[torch]'.
"""
from __future__ import annotations

from ..exceptions import BackendError


def require_torch() -> None:
    try:
        import torch  # noqa: F401, PLC0415
    except ImportError as exc:
        raise BackendError("H-Ngram layer requires torch: pip install 'atomicng[torch]'") from exc


__all__ = ["require_torch"]
