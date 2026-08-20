"""Leakage-safe dataset construction for directional research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .features import directional_consistency, rolling_volatility
from .labeling import PricePoint, build_outcome


@dataclass(frozen=True, init=False)
class ResearchRow:
    epoch: int
    quote: float
    return_1: float | None
    momentum_2: float | None
    volatility_5: float | None
    label: str | None
    actual_horizon_seconds: float | None
    ema_distance_10: float | None
    directional_consistency_5: float | None

    def __init__(self, epoch: int, quote: float, return_1: float | None, momentum_2: float | None, volatility_5: float | None, *args: Any, **kwargs: Any) -> None:
        fields = ("label", "actual_horizon_seconds", "ema_distance_10", "directional_consistency_5")
        if kwargs:
            if args or set(kwargs) - set(fields):
                raise TypeError("ResearchRow accepts either positional or named optional fields")
            values = {field: kwargs.get(field) for field in fields}
        elif len(args) == 2:
            values = {"label": args[0], "actual_horizon_seconds": args[1], "ema_distance_10": None, "directional_consistency_5": None}
        elif len(args) == 4:
            first, second, third, fourth = args
            if isinstance(third, str) or third is None:
                values = {"ema_distance_10": first, "directional_consistency_5": second, "label": third, "actual_horizon_seconds": fourth}
            else:
                values = {"label": first, "actual_horizon_seconds": second, "ema_distance_10": third, "directional_consistency_5": fourth}
        else:
            raise TypeError("ResearchRow expects 7 or 9 positional arguments")
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "quote", quote)
        object.__setattr__(self, "return_1", return_1)
        object.__setattr__(self, "momentum_2", momentum_2)
        object.__setattr__(self, "volatility_5", volatility_5)
        for field in fields:
            object.__setattr__(self, field, values[field])


def build_dataset(ticks: list[dict[str, Any]], horizon_seconds: int = 60) -> list[ResearchRow]:
    """Build rows using only current/past prices for features and future price for target."""
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    ordered = sorted(ticks, key=lambda tick: int(tick["epoch"]))
    prices = [float(tick["quote"]) for tick in ordered]
    epochs = [int(tick["epoch"]) for tick in ordered]
    points = [PricePoint(epoch=epoch, quote=price) for epoch, price in zip(epochs, prices)]
    rows: list[ResearchRow] = []
    ema_value: float | None = None
    alpha = 2.0 / 11.0
    for i, (epoch, price) in enumerate(zip(epochs, prices)):
        history_length = i + 1
        return_1 = price / prices[i - 1] - 1.0 if history_length >= 2 else None
        momentum_2 = price / prices[i - 2] - 1.0 if history_length >= 3 else None
        recent5 = prices[i - 4 : i + 1] if history_length >= 5 else []
        volatility_5 = rolling_volatility(recent5, 5) if recent5 else None
        directional_consistency_5 = directional_consistency(recent5, 5) if recent5 else None
        if history_length == 10:
            ema_value = sum(prices[:10]) / 10.0
        elif history_length > 10 and ema_value is not None:
            ema_value = alpha * price + (1.0 - alpha) * ema_value
        ema_distance_10 = price / ema_value - 1.0 if ema_value is not None and ema_value != 0 else None
        outcome = build_outcome(points, i, horizon_seconds)
        actual_horizon = outcome.future_epoch - outcome.observation_epoch if outcome.future_epoch is not None else None
        rows.append(ResearchRow(epoch=epoch, quote=price, return_1=return_1, momentum_2=momentum_2, volatility_5=volatility_5, label=outcome.direction, actual_horizon_seconds=actual_horizon, ema_distance_10=ema_distance_10, directional_consistency_5=directional_consistency_5))
    return rows


def temporal_split(rows: list[ResearchRow], train_ratio: float = 0.7) -> tuple[list[ResearchRow], list[ResearchRow]]:
    """Split chronologically; never shuffle observations across time."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    ordered = sorted(rows, key=lambda row: row.epoch)
    cut = int(len(ordered) * train_ratio)
    return ordered[:cut], ordered[cut:]
