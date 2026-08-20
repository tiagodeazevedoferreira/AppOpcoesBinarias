from __future__ import annotations

import bisect
import math
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class VolatilityLabelReport:
    multiplier: float
    rows: int
    train_rows: int
    test_rows: int
    flat_ratio: float
    persistence_accuracy: float
    balanced_accuracy: float
    rise_recall: float
    fall_recall: float
    flat_recall: float
    non_overlapping_persistence_accuracy: float | None
    non_overlapping_balanced_accuracy: float | None
    non_overlapping_rows: int


def _rolling_sigma(returns: list[float], end: int, window: int) -> float | None:
    if end < window:
        return None
    values = returns[end - window : end]
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    sigma = math.sqrt(variance)
    return sigma if sigma > 0 else None


def _label(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "RISE"
    if delta < -threshold:
        return "FALL"
    return "FLAT"


def _evaluate_labels(labels: list[str]) -> tuple[float, float, float, float, float]:
    if not labels:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    counts = Counter(labels)
    correct = Counter()
    previous = None
    for label in labels:
        if previous is not None and previous == label:
            correct[label] += 1
        previous = label
    persistence_correct = sum(correct.values())
    recalls = {
        label: correct[label] / counts[label] if counts[label] else 0.0
        for label in ("RISE", "FALL", "FLAT")
    }
    return (
        persistence_correct / len(labels),
        sum(recalls.values()) / 3.0,
        recalls["RISE"],
        recalls["FALL"],
        recalls["FLAT"],
    )


def evaluate_volatility_labels(
    ticks: list[dict[str, object]],
    *,
    horizon_seconds: int,
    train_ratio: float,
    multipliers: tuple[float, ...],
    volatility_window: int = 60,
) -> tuple[VolatilityLabelReport, ...]:
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if volatility_window <= 1:
        raise ValueError("volatility_window must be greater than 1")
    if not multipliers or any(value <= 0 for value in multipliers):
        raise ValueError("multipliers must be positive and non-empty")

    ordered = sorted(
        ((int(item["epoch"]), float(item["quote"])) for item in ticks),
        key=lambda item: item[0],
    )
    epochs = [item[0] for item in ordered]
    prices = [item[1] for item in ordered]
    returns = [0.0]
    returns.extend(
        prices[i] / prices[i - 1] - 1.0 if prices[i - 1] else 0.0
        for i in range(1, len(prices))
    )

    per_multiplier: list[VolatilityLabelReport] = []
    for multiplier in multipliers:
        labels: list[str | None] = []
        for i, epoch in enumerate(epochs):
            future_index = bisect.bisect_left(epochs, epoch + horizon_seconds, i + 1)
            if future_index >= len(epochs):
                labels.append(None)
                continue
            sigma = _rolling_sigma(returns, i, volatility_window)
            if sigma is None:
                labels.append(None)
                continue
            threshold = multiplier * sigma * math.sqrt(horizon_seconds)
            future_return = prices[future_index] / prices[i] - 1.0 if prices[i] else 0.0
            labels.append(_label(future_return, threshold))

        usable = [label for label in labels if label is not None]
        cut = int(len(labels) * train_ratio)
        train_labels = [label for label in labels[:cut] if label is not None]
        test_labels = [label for label in labels[cut:] if label is not None]
        persistence_accuracy, balanced, rise, fall, flat = _evaluate_labels(
            [*([train_labels[-1]] if train_labels else []), *test_labels]
        )

        selected: list[str] = []
        last_epoch: int | None = None
        for i, epoch in enumerate(epochs):
            label = labels[i]
            if label is None:
                continue
            if last_epoch is None or epoch >= last_epoch + horizon_seconds:
                selected.append(label)
                last_epoch = epoch
        non_cut = int(len(selected) * train_ratio)
        non_train = selected[:non_cut]
        non_test = selected[non_cut:]
        non_persistence, non_balanced, _, _, _ = _evaluate_labels(
            [*([non_train[-1]] if non_train else []), *non_test]
        )

        per_multiplier.append(
            VolatilityLabelReport(
                multiplier=multiplier,
                rows=len(usable),
                train_rows=len(train_labels),
                test_rows=len(test_labels),
                flat_ratio=sum(label == "FLAT" for label in usable) / len(usable) if usable else 0.0,
                persistence_accuracy=persistence_accuracy,
                balanced_accuracy=balanced,
                rise_recall=rise,
                fall_recall=fall,
                flat_recall=flat,
                non_overlapping_persistence_accuracy=non_persistence if non_test else None,
                non_overlapping_balanced_accuracy=non_balanced if non_test else None,
                non_overlapping_rows=len(selected),
            )
        )
    return tuple(per_multiplier)
