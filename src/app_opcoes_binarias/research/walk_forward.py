from __future__ import annotations

from dataclasses import dataclass

from .baselines import ClassificationMetrics, accuracy
from .dataset import ResearchRow, temporal_split
from .evaluation import evaluate_baselines
from .model_evaluation import evaluate_nearest_centroid, evaluate_softmax


@dataclass(frozen=True)
class FoldReport:
    train_rows: int
    test_rows: int
    baseline_majority: ClassificationMetrics
    baseline_persistence: ClassificationMetrics
    nearest_centroid: ClassificationMetrics
    softmax: ClassificationMetrics


@dataclass(frozen=True)
class WalkForwardReport:
    folds: tuple[FoldReport, ...]
    majority: ClassificationMetrics
    persistence: ClassificationMetrics
    nearest_centroid: ClassificationMetrics
    softmax: ClassificationMetrics


def _aggregate(metrics: list[ClassificationMetrics]) -> ClassificationMetrics:
    total = sum(item.total for item in metrics)
    correct = sum(item.correct for item in metrics)
    return ClassificationMetrics(
        accuracy=correct / total if total else 0.0,
        total=total,
        correct=correct,
    )


def evaluate_walk_forward(
    rows: list[ResearchRow],
    *,
    folds: int = 5,
    train_ratio: float = 0.7,
) -> WalkForwardReport:
    """Evaluate sequential expanding-window folds without future leakage."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    ordered = sorted(rows, key=lambda row: row.epoch)
    if len(ordered) < folds + 2:
        raise ValueError("not enough rows for requested walk-forward folds")

    fold_reports: list[FoldReport] = []
    majority_metrics: list[ClassificationMetrics] = []
    persistence_metrics: list[ClassificationMetrics] = []
    centroid_metrics: list[ClassificationMetrics] = []
    softmax_metrics: list[ClassificationMetrics] = []

    step = max(1, len(ordered) // (folds + 1))
    for fold_index in range(1, folds + 1):
        test_start = step * fold_index
        test_end = step * (fold_index + 1) if fold_index < folds else len(ordered)
        train = ordered[:test_start]
        test = ordered[test_start:test_end]
        if not train or not test:
            continue
        train, _ = temporal_split(train, train_ratio=1.0 - 1.0 / len(train))
        baseline = evaluate_baselines(train, test)
        centroid = evaluate_nearest_centroid(train, test)
        softmax = evaluate_softmax(train, test)
        report = FoldReport(
            train_rows=len(train),
            test_rows=len(test),
            baseline_majority=baseline.majority,
            baseline_persistence=baseline.persistence,
            nearest_centroid=centroid.model,
            softmax=softmax.model,
        )
        fold_reports.append(report)
        majority_metrics.append(baseline.majority)
        persistence_metrics.append(baseline.persistence)
        centroid_metrics.append(centroid.model)
        softmax_metrics.append(softmax.model)

    if not fold_reports:
        raise ValueError("no valid walk-forward folds were produced")
    return WalkForwardReport(
        folds=tuple(fold_reports),
        majority=_aggregate(majority_metrics),
        persistence=_aggregate(persistence_metrics),
        nearest_centroid=_aggregate(centroid_metrics),
        softmax=_aggregate(softmax_metrics),
    )
