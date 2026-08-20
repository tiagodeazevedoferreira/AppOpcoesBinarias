"""Deterministic future-outcome labeling for short-horizon research.

The labeler deliberately does not know anything about trading execution or
payouts. It answers only the research question: did the future price rise,
fall, or remain unchanged relative to the observation price?
"""

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PricePoint:
    epoch: float
    quote: float


@dataclass(frozen=True)
class Outcome:
    observation_epoch: float
    observation_quote: float
    horizon_seconds: int
    future_epoch: float | None
    future_quote: float | None
    direction: str | None


def label_direction(start: float, end: float, tolerance: float = 0.0) -> str:
    """Return RISE, FALL, or FLAT for two prices."""
    if end > start + tolerance:
        return "RISE"
    if end < start - tolerance:
        return "FALL"
    return "FLAT"


def find_future_point(
    points: Sequence[PricePoint],
    start_index: int,
    horizon_seconds: int,
) -> PricePoint | None:
    """Find the first point at or after the requested future timestamp."""
    if start_index < 0 or start_index >= len(points):
        raise IndexError("start_index is outside points")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")

    target = points[start_index].epoch + horizon_seconds
    epochs = [point.epoch for point in points]
    future_index = bisect_left(epochs, target, lo=start_index + 1)
    if future_index >= len(points):
        return None
    return points[future_index]


def build_outcome(
    points: Sequence[PricePoint],
    start_index: int,
    horizon_seconds: int = 60,
    tolerance: float = 0.0,
) -> Outcome:
    """Create one leakage-safe future outcome from an ordered tick series."""
    if not points:
        raise ValueError("points cannot be empty")

    start = points[start_index]
    future = find_future_point(points, start_index, horizon_seconds)
    if future is None:
        return Outcome(
            observation_epoch=start.epoch,
            observation_quote=start.quote,
            horizon_seconds=horizon_seconds,
            future_epoch=None,
            future_quote=None,
            direction=None,
        )

    return Outcome(
        observation_epoch=start.epoch,
        observation_quote=start.quote,
        horizon_seconds=horizon_seconds,
        future_epoch=future.epoch,
        future_quote=future.quote,
        direction=label_direction(start.quote, future.quote, tolerance),
    )
