"""Leakage-safe dataset construction for directional research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .features import (
    directional_consistency,
    ema_distance,
    momentum,
    returns,
    rolling_volatility,
)
from .labeling import PricePoint, build_outcome


@dataclass(frozen=True)
class ResearchRow:
    epoch: int
    quote: float
    return_1: float | None
    momentum_2: float | None
    volatility_5: float | None
    label: str | None
    actual_horizon_seconds: float | None
    ema_distance_10: float | None = None
    directional_consistency_5: float | None = None


def build_dataset(ticks: list[dict[str, Any]], horizon_seconds: int = 60) -> list[ResearchRow]:
    """Build rows using only current/past prices for features and future price for target."""
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    ordered = sorted(ticks, key=lambda tick: int(tick["epoch"]))
    prices = [float(tick["quote"]) for tick in ordered]
    epochs = [int(tick["epoch"]) for tick in ordered]
    points = [PricePoint(epoch=epoch, quote=price) for epoch, price in zip(epochs, prices)]
    rows: list[ResearchRow] = []

    for i, (epoch, price) in enumerate(zip(epochs, prices)):
        past = prices[: i + 1]
        outcome = build_outcome(points, i, horizon_seconds)
        actual_horizon = (
            outcome.future_epoch - outcome.observation_epoch
            if outcome.future_epoch is not None
            else None
        )
        rows.append(
            ResearchRow(
                epoch=epoch,
                quote=price,
                return_1=returns(past)[-1] if len(past) >= 2 else None,
                momentum_2=momentum(past, 2),
                volatility_5=rolling_volatility(past, 5),
                ema_distance_10=ema_distance(past, 10),
                directional_consistency_5=directional_consistency(past, 5),
                label=outcome.direction,
                actual_horizon_seconds=actual_horizon,
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
