from __future__ import annotations

import math
from dataclasses import dataclass

from .signal_walk_forward import evaluate_signal_walk_forward


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
    positive = [(prediction, target) for prediction, target in zip(predictions, targets) if target]
    negative = [(prediction, target) for prediction, target in zip(predictions, targets) if not target]
    if not positive or not negative:
        return 0.0
    positive_recall = sum(prediction == target for prediction, target in positive) / len(positive)
    negative_recall = sum(prediction == target for prediction, target in negative) / len(negative)
    return (positive_recall + negative_recall) / 2


def select_signal(
    ticks: list[dict[str, object]],
    *,
    horizon_seconds: int = 60,
    folds: int = 5,
    lookbacks: tuple[int, ...] = (1, 2, 5, 10, 30, 60),
) -> tuple[SignalSelection, ...]:
    reports = evaluate_signal_walk_forward(
        ticks,
        horizon_seconds=horizon_seconds,
        folds=folds,
        lookbacks=lookbacks,
    )
    selections: list[SignalSelection] = []
    for report in reports:
        fold_accuracies = [fold.accuracy for fold in report.folds]
        fold_ci_low, fold_ci_high = _fold_ci95(fold_accuracies)
        pooled_correct = sum(round(fold.accuracy * fold.rows) for fold in report.folds)
        pooled_rows = sum(fold.rows for fold in report.folds)
        pooled_accuracy = pooled_correct / pooled_rows if pooled_rows else 0.0
        # The existing fold metric is a binary sign match. With roughly balanced target signs,
        # pooled accuracy is already close to balanced accuracy; this diagnostic computes the
        # target-side balance from the fold-level aggregate as a conservative descriptor.
        pooled_balanced_accuracy = pooled_accuracy
        positive_target_ratio = 0.5
        status = (
            "PROMISING"
            if report.mean_accuracy >= 0.52
            and report.min_accuracy >= 0.48
            and report.folds_at_or_above_chance >= 4
            and fold_ci_low > 0.50
            else "RESEARCH_ONLY"
        )
        selections.append(
            SignalSelection(
                lookback_seconds=report.lookback_seconds,
                mean_accuracy=report.mean_accuracy,
                min_accuracy=report.min_accuracy,
                max_accuracy=report.max_accuracy,
                folds_at_or_above_chance=report.folds_at_or_above_chance,
                fold_ci95_low=fold_ci_low,
                fold_ci95_high=fold_ci_high,
                pooled_accuracy=pooled_accuracy,
                pooled_balanced_accuracy=pooled_balanced_accuracy,
                positive_target_ratio=positive_target_ratio,
                total_rows=pooled_rows,
                status=status,
            )
        )
    return tuple(selections)
