"""Chronological dataset splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TemporalSplit:
    train: list[T]
    test: list[T]


def chronological_split(items: Sequence[T], train_fraction: float = 0.8) -> TemporalSplit:
    """Split ordered observations without shuffling or leaking future data."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if len(items) < 2:
        raise ValueError("at least two observations are required")

    cut = int(len(items) * train_fraction)
    cut = max(1, min(cut, len(items) - 1))
    return TemporalSplit(train=list(items[:cut]), test=list(items[cut:]))
