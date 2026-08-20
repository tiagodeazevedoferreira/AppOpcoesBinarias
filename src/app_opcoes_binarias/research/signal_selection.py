from __future__ import annotations

import bisect
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SignalSelection:
    lookback_seconds: int
    mean_accuracy: float
    min_accuracy: float
    max_accuracy: float
    folds_at_or_above_chance: int
    fold_ci95_low: float
    fold_ci95_high: float
    pooled_accuracy: float
    pooled_balanced_accuracy: float
    positive_target_ratio: float
    total_rows: int
    status: str


def _fold_ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    se = math.sqrt(variance / len(values))
    margin = 2.7764451051977987 * se
    return max(0.0, mean - margin), min(1.0, mean + margin)


def _balanced_accuracy(predictions: list[bool], targets: list[bool]) -> float:
    if not predictions or len(predictions) != len(targets):
        return 0.0
    positive_total = sum(targets)
    negative_total = len(targets) - positive_total
    if not positive_total or not negative_total:
        return 0.0
    positive_recall = sum(prediction and target for prediction, target in zip(predictions, targets, strict=True)) / positive_total
    negative_recall = sum((not prediction) and (not target) for prediction, target in zip(predictions, targets, strict=True)) / negative_total
    return (positive_recall + negative_recall) / 2


def select_signal(
    ticks: list[dict[str, object]],
    *,
    horizon_seconds: int = 60,
    folds: int = 5,
    lookbacks: tuple[int, ...] = (1, 2, 5, 10, 30, 60),
) -> tuple[SignalSelection, ...]:
    if horizon_seconds <= 0 or folds < 2:
        raise ValueError("horizon_seconds must be positive and folds must be at least 2")

    ordered = sorted(
        ((int(item["epoch"]), float(item["quote"])) for item in ticks),
        key=lambda item: item[0],
    )
    epochs = [item[0] for item in ordered]
    prices = [item[1] for item in ordered]
    selections: list[SignalSelection] = []

    for lookback in lookbacks:
        if lookback <= 0:
            raise ValueError("lookbacks must be positive")

        pairs: list[tuple[int, bool, bool]] = []
        for i, epoch in enumerate(epochs):
            if i < lookback or prices[i] == 0 or prices[i - lookback] == 0:
                continue
            future = bisect.bisect_left(epochs, epoch + horizon_seconds, i + 1)
            if future >= len(epochs):
                continue
            feature = prices[i] / prices[i - lookback] - 1.0
            target = prices[future] / prices[i] - 1.0
            if feature == 0 or target == 0:
                continue
            pairs.append((epoch, feature > 0, target > 0))

        selected: list[tuple[int, bool, bool]] = []
        last_epoch: int | None = None
        for row in pairs:
            if last_epoch is None or row[0] >= last_epoch + horizon_seconds:
                selected.append(row)
                last_epoch = row[0]

        fold_size = len(selected) // folds if selected else 0
        fold_accuracies: list[float] = []
        fold_rows: list[list[tuple[int, bool, bool]]] = []
        for fold in range(folds):
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold < folds - 1 else len(selected)
            test = selected[start:end]
            fold_rows.append(test)
            fold_accuracies.append(
                sum(prediction == target for _, prediction, target in test) / len(test) if test else 0.0
            )

        pooled = [row for fold in fold_rows for row in fold]
        predictions = [prediction for _, prediction, _ in pooled]
        targets = [target for _, _, target in pooled]
        pooled_accuracy = sum(prediction == target for prediction, target in zip(predictions, targets, strict=True)) / len(pooled) if pooled else 0.0
        pooled_balanced = _balanced_accuracy(predictions, targets)
        positive_ratio = sum(targets) / len(targets) if targets else 0.0
        ci_low, ci_high = _fold_ci95(fold_accuracies)
        mean_accuracy = sum(fold_accuracies) / len(fold_accuracies) if fold_accuracies else 0.0
        status = (
            "PROMISING"
            if fold_accuracies
            and sum(value >= 0.5 for value in fold_accuracies) >= 4
            and mean_accuracy >= 0.52
            and min(fold_accuracies) >= 0.48
            and ci_low > 0.50
            and pooled_balanced >= 0.50
            else "RESEARCH_ONLY"
        )
        selections.append(
            SignalSelection(
                lookback_seconds=lookback,
                mean_accuracy=mean_accuracy,
                min_accuracy=min(fold_accuracies) if fold_accuracies else 0.0,
                max_accuracy=max(fold_accuracies) if fold_accuracies else 0.0,
                folds_at_or_above_chance=sum(value >= 0.5 for value in fold_accuracies),
                fold_ci95_low=ci_low,
                fold_ci95_high=ci_high,
                pooled_accuracy=pooled_accuracy,
                pooled_balanced_accuracy=pooled_balanced,
                positive_target_ratio=positive_ratio,
                total_rows=len(pooled),
                status=status,
            )
        )

    return tuple(selections)
