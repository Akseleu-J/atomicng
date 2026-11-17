"""Write-rule interfaces. WriteRule.apply(patch, events) returns a WriteReport."""

from .d1 import D1Rule, Event, RowReport, WriteReport

__all__ = ["D1Rule", "Event", "RowReport", "WriteReport"]
