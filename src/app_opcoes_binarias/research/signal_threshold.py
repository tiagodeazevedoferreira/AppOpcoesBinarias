from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdFold:
    fold: int
    threshold: float
    test_rows: int
    directional_rows: int
    accuracy: float
    decision_rate: float


@dataclass(frozen=True)
class ThresholdReport:
    threshold_quantile: float
    folds: tuple[ThresholdFold, ...]
    mean_accuracy: float
    min_accuracy: float
    max_accuracy: float
    mean_decision_rate: float
    total_directional_rows: int


def evaluate_signal_thresholds(
    ticks: list[dict[str, object]],
    *,
    horizon_seconds: int = 60,
    lookback_seconds: int = 60,
    folds: int = 20,
    quantiles: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90),
) -> tuple[ThresholdReport, ...]:
    if horizon_seconds <= 0 or lookback_seconds <= 0 or lookback_seconds > horizon_seconds:
        raise ValueError("lookback_seconds must be positive and at most horizon_seconds")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if not quantiles or any(value <= 0 or value >= 1 for value in quantiles):
        raise ValueError("quantiles must be between 0 and 1")

    ordered = sorted(
        ((int(item["epoch"]), float(item["quote"])) for item in ticks),
        key=lambda item: item[0],
    )
    epochs = [item[0] for item in ordered]
    prices = [item[1] for item in ordered]
    pairs: list[tuple[int, float, float]] = []

    for i, epoch in enumerate(epochs):
        lookback_index = bisect.bisect_right(epochs, epoch - lookback_seconds, 0, i + 1) - 1
        if lookback_index < 0 or prices[i] == 0 or prices[lookback_index] == 0:
            continue
        future_index = bisect.bisect_left(epochs, epoch + horizon_seconds, i + 1)
        if future_index >= len(epochs) or prices[future_index] == 0:
            continue
        feature = prices[i] / prices[lookback_index] - 1.0
        target = prices[future_index] / prices[i] - 1.0
        if feature == 0 or target == 0:
            continue
        pairs.append((epoch, feature, target))

    selected: list[tuple[int, float, float]] = []
    last_epoch: int | None = None
    for row in pairs:
        if last_epoch is None or row[0] >= last_epoch + horizon_seconds:
            selected.append(row)
            last_epoch = row[0]

    fold_size = len(selected) // folds
    reports: list[ThresholdReport] = []
    for quantile in quantiles:
        fold_reports: list[ThresholdFold] = []
        for fold in range(folds):
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold < folds - 1 else len(selected)
            test = selected[start:end]
            train = selected[:start]
            magnitudes = sorted(abs(feature) for _, feature, _ in train)
            if not magnitudes:
                continue
            index = min(len(magnitudes) - 1, int(quantile * (len(magnitudes) - 1)))
            threshold = magnitudes[index]
            directional = [row for row in test if abs(row[1]) >= threshold]
            correct = sum((feature > 0) == (target > 0) for _, feature, target in directional)
            directional_rows = len(directional)
            test_rows = len(test)
            fold_reports.append(
                ThresholdFold(
                    fold=fold + 1,
                    threshold=threshold,
                    test_rows=test_rows,
                    directional_rows=directional_rows,
                    accuracy=correct / directional_rows if directional_rows else 0.0,
                    decision_rate=directional_rows / test_rows if test_rows else 0.0,
                )
            )

        if not fold_reports:
            reports.append(ThresholdReport(quantile, tuple(), 0.0, 0.0, 0.0, 0.0, 0))
            continue
        accuracies = [item.accuracy for item in fold_reports]
        decision_rates = [item.decision_rate for item in fold_reports]
        reports.append(
            ThresholdReport(
                threshold_quantile=quantile,
                folds=tuple(fold_reports),
                mean_accuracy=sum(accuracies) / len(accuracies),
                min_accuracy=min(accuracies),
                max_accuracy=max(accuracies),
                mean_decision_rate=sum(decision_rates) / len(decision_rates),
                total_directional_rows=sum(item.directional_rows for item in fold_reports),
            )
        )
    return tuple(reports)
