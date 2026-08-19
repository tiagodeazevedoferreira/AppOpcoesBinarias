"""Small, deterministic feature functions for the first research baseline."""

from __future__ import annotations

from collections.abc import Sequence
import math


def returns(prices: Sequence[float], lag: int = 1) -> list[float]:
    """Calculate simple percentage returns without looking into the future."""
    if lag <= 0:
        raise ValueError("lag must be positive")
    if len(prices) <= lag:
        return []
    return [prices[i] / prices[i - lag] - 1.0 for i in range(lag, len(prices))]


def rolling_volatility(prices: Sequence[float], window: int) -> float | None:
    """Population standard deviation of simple returns in the trailing window."""
    if window <= 1 or len(prices) < window + 1:
        return None
    values = returns(prices[-(window + 1) :])
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def momentum(prices: Sequence[float], lookback: int) -> float | None:
    """Trailing price momentum as a fractional change."""
    if lookback <= 0 or len(prices) <= lookback:
        return None
    previous = prices[-lookback - 1]
    if previous == 0:
        return None
    return prices[-1] / previous - 1.0


def ema(prices: Sequence[float], period: int) -> float | None:
    """Exponential moving average using only observations supplied in prices."""
    if period <= 0 or len(prices) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    value = sum(prices[:period]) / period
    for price in prices[period:]:
        value = alpha * price + (1.0 - alpha) * value
    return value
