"""Правила записи (плагин-слой): контракт WriteRule.apply(patch, events) → WriteReport."""

from .d1 import D1Rule, Event, RowReport, WriteReport

__all__ = ["D1Rule", "Event", "RowReport", "WriteReport"]
