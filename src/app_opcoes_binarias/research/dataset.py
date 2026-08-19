"""Leakage-safe dataset construction for directional research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .features import momentum, returns, rolling_volatility
from .labeling import label_60s


@dataclass(frozen=True)
class ResearchRow:
    epoch: int
    quote: float
    return_1: float | None
    momentum_2: float | None
    volatility_5: float | None
    label: str | None


def build_dataset(ticks: list[dict[str, Any]]) -> list[ResearchRow]:
    """Build rows using only current/past prices for features and future price for label."""
    ordered = sorted(ticks, key=lambda tick: int(tick["epoch"]))
    prices = [float(tick["quote"]) for tick in ordered]
    epochs = [int(tick["epoch"]) for tick in ordered]
    rows: list[ResearchRow] = []

    for i, (epoch, price) in enumerate(zip(epochs, prices)):
        past = prices[: i + 1]
        future = next((p for e, p in zip(epochs, prices) if e >= epoch + 60), None)
        rows.append(
            ResearchRow(
                epoch=epoch,
                quote=price,
                return_1=returns(past)[-1] if len(past) >= 2 else None,
                momentum_2=momentum(past, 2),
                volatility_5=rolling_volatility(past, 5),
                label=label_60s(price, future),
            )
        )
    return rows


def temporal_split(rows: list[ResearchRow], train_ratio: float = 0.7) -> tuple[list[ResearchRow], list[ResearchRow]]:
    """Split chronologically; never shuffle observations across time."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    ordered = sorted(rows, key=lambda row: row.epoch)
    cut = int(len(ordered) * train_ratio)
    return ordered[:cut], ordered[cut:]
