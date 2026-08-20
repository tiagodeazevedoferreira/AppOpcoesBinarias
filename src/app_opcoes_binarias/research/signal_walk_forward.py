from __future__ import annotations

import bisect
from dataclasses import dataclass

from .signal_diagnostics import evaluate_signal_diagnostics


@dataclass(frozen=True)
class SignalFold:
    fold: int
    lookback_seconds: int
    rows: int
    accuracy: float


@dataclass(frozen=True)
class SignalWalkForwardReport:
    lookback_seconds: int
    folds: tuple[SignalFold, ...]
    mean_accuracy: float
    min_accuracy: float
    max_accuracy: float
    folds_at_or_above_chance: int


def evaluate_signal_walk_forward(
    ticks: list[dict[str, object]],
    *,
    horizon_seconds: int = 60,
    folds: int = 5,
    lookbacks: tuple[int, ...] = (1, 2, 5, 10, 30, 60),
) -> tuple[SignalWalkForwardReport, ...]:
    if horizon_seconds <= 0 or folds < 2:
        raise ValueError("horizon_seconds must be positive and folds must be at least 2")
    if not lookbacks or any(value <= 0 for value in lookbacks):
        raise ValueError("lookbacks must be positive and non-empty")

    ordered = sorted(
        ((int(item["epoch"]), float(item["quote"])) for item in ticks),
        key=lambda item: item[0],
    )
    epochs = [item[0] for item in ordered]
    prices = [item[1] for item in ordered]
    reports: list[SignalWalkForwardReport] = []

    for lookback in lookbacks:
        pairs: list[tuple[int, float, float]] = []
        for i, epoch in enumerate(epochs):
            lookback_index = bisect.bisect_right(epochs, epoch - lookback, 0, i + 1) - 1
            if lookback_index < 0:
                continue
            future = bisect.bisect_left(epochs, epoch + horizon_seconds, i + 1)
            if future >= len(epochs) or prices[i] == 0 or prices[lookback_index] == 0:
                continue
            feature = prices[i] / prices[lookback_index] - 1.0
            target = prices[future] / prices[i] - 1.0
            pairs.append((epoch, feature, target))

        selected: list[tuple[int, float, float]] = []
        last_epoch: int | None = None
        for row in pairs:
            if last_epoch is None or row[0] >= last_epoch + horizon_seconds:
                selected.append(row)
                last_epoch = row[0]

        if not selected:
            reports.append(SignalWalkForwardReport(lookback, tuple(), 0.0, 0.0, 0.0, 0))
            continue

        fold_size = len(selected) // folds
        fold_reports: list[SignalFold] = []
        for fold in range(folds):
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold < folds - 1 else len(selected)
            test = selected[start:end]
            correct = sum(
                (feature > 0) == (target > 0)
                for _, feature, target in test
                if feature != 0 and target != 0
            )
            usable = sum(feature != 0 and target != 0 for _, feature, target in test)
            accuracy = correct / usable if usable else 0.0
            fold_reports.append(SignalFold(fold + 1, lookback, usable, accuracy))

        accuracies = [row.accuracy for row in fold_reports]
        reports.append(
            SignalWalkForwardReport(
                lookback_seconds=lookback,
                folds=tuple(fold_reports),
                mean_accuracy=sum(accuracies) / len(accuracies),
                min_accuracy=min(accuracies),
                max_accuracy=max(accuracies),
                folds_at_or_above_chance=sum(value >= 0.5 for value in accuracies),
            )
        )

    evaluate_signal_diagnostics(ticks, horizon_seconds=horizon_seconds, lookbacks=lookbacks)
    return tuple(reports)
