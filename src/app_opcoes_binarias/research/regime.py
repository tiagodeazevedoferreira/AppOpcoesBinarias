from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeLabel:
    start_epoch: int
    end_epoch: int
    return_ratio: float
    volatility: float
    regime: str


def label_regime(epochs: list[int], quotes: list[float], *, window: int = 60) -> RegimeLabel | None:
    """Classify a recent price window as trend, reversal-prone, or quiet."""
    if window <= 0 or len(epochs) < window + 1 or len(quotes) != len(epochs):
        return None
    start_quote = float(quotes[-window - 1])
    end_quote = float(quotes[-1])
    if start_quote == 0:
        return None
    returns = [abs(float(quotes[i]) / float(quotes[i - 1]) - 1.0) for i in range(1, len(quotes))]
    recent = returns[-window:]
    mean_abs_return = sum(recent) / len(recent)
    total_return = end_quote / start_quote - 1.0
    if mean_abs_return == 0.0:
        regime = "QUIET"
    elif abs(total_return) > 2.0 * mean_abs_return * window**0.5:
        regime = "TREND"
    else:
        regime = "NOISY"
    return RegimeLabel(
        start_epoch=int(epochs[-window - 1]),
        end_epoch=int(epochs[-1]),
        return_ratio=total_return,
        volatility=mean_abs_return,
        regime=regime,
    )
