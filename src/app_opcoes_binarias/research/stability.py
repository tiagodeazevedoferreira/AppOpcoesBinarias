from __future__ import annotations

import math
from dataclasses import dataclass

from .baselines import ClassificationMetrics
from .walk_forward import FoldReport, WalkForwardReport


@dataclass(frozen=True)
class StabilityMetrics:
    accuracy: float
    folds: int
    folds_at_or_above_chance: int
    folds_better_than_persistence: int
    min_fold_accuracy: float
    max_fold_accuracy: float
    mean_fold_accuracy: float
    chance_gap: float
    persistence_gap: float
    naive_ci95_low: float
    naive_ci95_high: float


@dataclass(frozen=True)
class StabilityReport:
    majority: StabilityMetrics
    persistence: StabilityMetrics
    nearest_centroid: StabilityMetrics
    softmax: StabilityMetrics


def _metrics(
    aggregate: ClassificationMetrics,
    folds: tuple[FoldReport, ...],
    selector: str,
) -> StabilityMetrics:
    values = [getattr(fold, selector).accuracy for fold in folds]
    if not values:
        raise ValueError("stability analysis requires at least one fold")
    mean_accuracy = sum(values) / len(values)
    n = aggregate.total
    p = aggregate.accuracy
    if n:
        margin = 1.96 * math.sqrt(max(0.0, p * (1.0 - p) / n))
        low = max(0.0, p - margin)
        high = min(1.0, p + margin)
    else:
        low = high = 0.0
    persistence_values = [fold.baseline_persistence.accuracy for fold in folds]
    better_than_persistence = sum(
        value > baseline for value, baseline in zip(values, persistence_values)
    )
    return StabilityMetrics(
        accuracy=aggregate.accuracy,
        folds=len(values),
        folds_at_or_above_chance=sum(value >= 0.5 for value in values),
        folds_better_than_persistence=better_than_persistence,
        min_fold_accuracy=min(values),
        max_fold_accuracy=max(values),
        mean_fold_accuracy=mean_accuracy,
        chance_gap=aggregate.accuracy - 0.5,
        persistence_gap=aggregate.accuracy - sum(persistence_values) / len(persistence_values),
        naive_ci95_low=low,
        naive_ci95_high=high,
    )


def evaluate_walk_forward_stability(report: WalkForwardReport) -> StabilityReport:
    """Summarize fold consistency without tuning on the test folds."""
    folds = report.folds
    return StabilityReport(
        majority=_metrics(report.majority, folds, "baseline_majority"),
        persistence=_metrics(report.persistence, folds, "baseline_persistence"),
        nearest_centroid=_metrics(report.nearest_centroid, folds, "nearest_centroid"),
        softmax=_metrics(report.softmax, folds, "softmax"),
    )
