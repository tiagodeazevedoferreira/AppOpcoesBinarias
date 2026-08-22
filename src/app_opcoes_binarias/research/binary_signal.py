from __future__ import annotations

import bisect
from dataclasses import dataclass
from math import floor

from .dataset import ResearchRow


@dataclass(frozen=True)
class BinarySignalReport:
    lookback_seconds: int
    quantile: float
    threshold: float
    accuracy: float
    correct: int
    decisions: int
    decision_rate: float


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot calculate quantile of empty values")
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _momentum(row: ResearchRow, epochs: list[int], quotes: list[float], lookback: int) -> float | None:
    index = bisect.bisect_right(epochs, row.epoch - lookback) - 1
    if index < 0:
        return None
    previous = quotes[index]
    if previous == 0.0 or row.quote == 0.0:
        return None
    return row.quote / previous - 1.0


def _usable_direction_rows(rows: list[ResearchRow]) -> list[ResearchRow]:
    return [row for row in rows if row.label in {"RISE", "FALL"}]


def evaluate_binary_momentum(
    train: list[ResearchRow],
    test: list[ResearchRow],
    *,
    lookback_seconds: int = 60,
    quantile: float = 0.0,
) -> BinarySignalReport:
    """Evaluate momentum sign only on directional outcomes, with train-only thresholding."""
    if lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be positive")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")

    combined = sorted(train + test, key=lambda row: row.epoch)
    epochs = [row.epoch for row in combined]
    quotes = [row.quote for row in combined]
    train_directional = _usable_direction_rows(train)
    train_magnitudes = []
    for row in train_directional:
        value = _momentum(row, epochs, quotes, lookback_seconds)
        if value is not None and value != 0.0:
            train_magnitudes.append(abs(value))
    threshold = _quantile(train_magnitudes, quantile) if train_magnitudes else 0.0

    correct = 0
    decisions = 0
    for row in _usable_direction_rows(test):
        value = _momentum(row, epochs, quotes, lookback_seconds)
        if value is None or value == 0.0 or abs(value) < threshold:
            continue
        prediction = "RISE" if value > 0.0 else "FALL"
        decisions += 1
        correct += int(prediction == row.label)

    return BinarySignalReport(
        lookback_seconds=lookback_seconds,
        quantile=quantile,
        threshold=threshold,
        accuracy=correct / decisions if decisions else 0.0,
        correct=correct,
        decisions=decisions,
        decision_rate=decisions / len(test) if test else 0.0,
    )


def evaluate_binary_momentum_walk_forward(
    rows: list[ResearchRow],
    *,
    lookback_seconds: int = 60,
    quantile: float = 0.7,
    folds: int = 5,
) -> tuple[BinarySignalReport, ...]:
    """Evaluate expanding-window binary momentum with a threshold fit on each train fold."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    ordered = sorted(rows, key=lambda row: row.epoch)
    if len(ordered) < folds + 2:
        raise ValueError("not enough rows for requested walk-forward folds")

    step = max(1, len(ordered) // (folds + 1))
    reports: list[BinarySignalReport] = []
    for fold_index in range(1, folds + 1):
        test_start = step * fold_index
        test_end = step * (fold_index + 1) if fold_index < folds else len(ordered)
        train = ordered[:test_start]
        test = ordered[test_start:test_end]
        if train and test:
            reports.append(
                evaluate_binary_momentum(
                    train,
                    test,
                    lookback_seconds=lookback_seconds,
                    quantile=quantile,
                )
            )
    if not reports:
        raise ValueError("no valid walk-forward folds were produced")
    return tuple(reports)
