"""SubtableManager — реестр L2-томов, LRU-слоты на устройстве и маршрутизация.

В RAM (на устройстве вычислений) держится не более n_slots субтаблиц; остальные
пребывают на «диске» (npz-файлы) и подгружаются при смене темы. Переключение
делает только роутер; менеджер обеспечивает горячую подмену и статистику:
время подгрузки, число эвикций, признак thrash.

Промах темы или отсутствие активного тома — деградация к чистому L1: значения
читаются как нули (hits=False), система не падает.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass

import torch

from .hashing import addresses
from .store import L2Subtable


@dataclass
class L2Read:
    """Результат одного чтения внешней памяти: активный том + собранные строки."""

    name: str | None                  # активная субтаблица (None — только L1)
    weight: float                     # вес активной темы (draft: 1.0)
    values: torch.Tensor              # (B, T-key_order+1, d) — Δ-вектора строки
    hits: torch.Tensor                # (B, T-key_order+1) bool
    switched: bool                    # тема сменилась на этом чтении
    load_ms: float                    # время последней подгрузки тома с «диска», мс
    evicted: str | None               # имя вытесненного тома (LRU), если было


class SubtableManager:
    """Реестр субтаблиц + LRU-слоты. Роутер передаётся снаружи (или None)."""

    def __init__(
        self,
        directory: str,
        n_slots: int = 2,
        device: torch.device | str | None = None,
        router=None,  # TopicRouter | None
    ) -> None:
        if n_slots < 1:
            raise ValueError("n_slots >= 1.")
        self.directory = directory
        self.n_slots = n_slots
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.router = router
        self.force_name: str | None = None   # ручная фиксация тома (pin): бypass роутера
        self._registry: dict[str, str] = {}          # name → путь npz («диск»)
        self._resident: OrderedDict[str, L2Subtable] = OrderedDict()
        os.makedirs(directory, exist_ok=True)
        self.stats = {"loads": 0, "evictions": 0, "total_load_ms": 0.0}

    # ----------------------------------------------------------- реестр

    def register(self, subtable: L2Subtable) -> None:
        """Сохранить том на диск и внести в реестр (холодное состояние)."""
        path = os.path.join(self.directory, f"{subtable.spec.name}.npz")
        subtable.save_pretrained(path)
        self._registry[subtable.spec.name] = path

    @property
    def names(self) -> list[str]:
        return list(self._registry)

    # ------------------------------------------------------- LRU-слоты

    def ensure(self, name: str) -> tuple[L2Subtable, str | None, float]:
        """(том, вытесненное имя, время подгрузки мс). Загрузка — с «диска»."""
        if name not in self._registry:
            raise KeyError(f"Subtable {name!r} is not registered.")
        if name in self._resident:
            self._resident.move_to_end(name)
            return self._resident[name], None, 0.0
        evicted = None
        while len(self._resident) >= self.n_slots:
            evicted, cold = self._resident.popitem(last=False)
            del cold
            self.stats["evictions"] += 1
        t0 = time.perf_counter()
        sub = L2Subtable.load_pretrained(self._registry[name], device=self.device)
        load_ms = (time.perf_counter() - t0) * 1000.0
        self.stats["loads"] += 1
        self.stats["total_load_ms"] += load_ms
        self._resident[name] = sub
        return sub, evicted, load_ms

    # --------------------------------------------------------- маршрут

    def pin(self, name: str | None) -> None:
        """Фиксировать том вручную (PoC/тесты/аварийный режим); None — отпустить."""
        if name is not None and name not in self._registry:
            raise KeyError(f"Subtable {name!r} is not registered.")
        self.force_name = name

    def route(self, features: torch.Tensor) -> tuple[str | None, float, object | None]:
        """Роутер → (имя активного тома, вес, решение). Без роутера — None."""
        if self.force_name is not None:
            return self.force_name, 1.0, None
        if self.router is None or not self._registry:
            return None, 0.0, None
        decision = self.router.route(features)
        return decision.name, 1.0, decision

    def read(self, features: torch.Tensor, input_ids: torch.Tensor, key_order: int = 2) -> L2Read:
        """Полный конвейер: маршрут → резидентность → gather адресов.

        input_ids: (B, T) токены; features: (B, T, d) признаки окна для роутера.
        Адресация — по схеме АКТИВНОЙ субтаблицы (инвариант: запись и чтение
        субтаблицы используют одну схему; key_order берётся из спецификации тома).
        """
        name, weight, decision = self.route(features)
        if name is None:
            w = max(input_ids.shape[1] - key_order + 1, 0)
            return L2Read(
                None, 0.0,
                torch.zeros((input_ids.shape[0], w, 0), device=self.device),
                torch.zeros((input_ids.shape[0], w), dtype=torch.bool, device=self.device),
                False, 0.0, None,
            )
        sub, evicted, load_ms = self.ensure(name)
        rows = addresses(input_ids, sub.spec.n_rows, sub.spec.key_order, sub.spec.hash_scheme, self.device)
        values, hits = sub.lookup(rows)
        switched = bool(decision.switched) if decision is not None else False
        return L2Read(name, weight, values, hits, switched, load_ms, evicted)

    def stats_summary(self) -> dict:
        """Сводка: слоты, подгрузки, среднее время подгрузки, thrash роутера."""
        loads = self.stats["loads"]
        r_stats = self.router.stats() if self.router is not None else {}
        return {
            "n_registered": len(self._registry),
            "n_resident": len(self._resident),
            "loads": loads,
            "evictions": self.stats["evictions"],
            "total_load_ms": round(self.stats["total_load_ms"], 3),
            "mean_load_ms": round(self.stats["total_load_ms"] / loads, 3) if loads else 0.0,
            "router": r_stats,
        }
