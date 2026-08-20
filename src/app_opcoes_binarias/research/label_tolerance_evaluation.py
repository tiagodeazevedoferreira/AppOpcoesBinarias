from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .baselines import persistence_prediction
from .dataset import ResearchRow, build_dataset, temporal_split
from .evaluation import evaluate_baselines, sample_non_overlapping


@dataclass(frozen=True)
class ClassificationShape:
    balanced_accuracy: float
    rise_recall: float
    fall_recall: float
    flat_recall: float
    test_distribution: dict[str, int]


@dataclass(frozen=True)
class ToleranceReport:
    tolerance: float
    rows: int
    train_rows: int
    test_rows: int
    flat_ratio: float
    persistence_accuracy: float
    non_overlapping_persistence_accuracy: float | None
    non_overlapping_rows: int
    persistence_shape: ClassificationShape
    non_overlapping_persistence_shape: ClassificationShape | None


def _shape(test: list[ResearchRow], train: list[ResearchRow]) -> ClassificationShape:
    labeled_train = [row.label for row in train if row.label is not None]
    labeled_test = [row for row in test if row.label is not None]
    previous: str | None = labeled_train[-1] if labeled_train else None
    counts = Counter(row.label for row in labeled_test)
    correct = Counter()
    for row in labeled_test:
        prediction = persistence_prediction(previous)
        if prediction == row.label:
            correct[row.label] += 1
        previous = row.label

    recalls = {
        label: (correct[label] / counts[label] if counts[label] else 0.0)
        for label in ("RISE", "FALL", "FLAT")
    }
    return ClassificationShape(
        balanced_accuracy=sum(recalls.values()) / 3.0,
        rise_recall=recalls["RISE"],
        fall_recall=recalls["FALL"],
        flat_recall=recalls["FLAT"],
        test_distribution=dict(counts),
    )


def evaluate_tolerance_sweep(
    ticks: list[dict[str, object]],
    *,
    horizon_seconds: int,
    train_ratio: float,
    tolerances: tuple[float, ...],
) -> tuple[ToleranceReport, ...]:
    if not tolerances:
        raise ValueError("tolerances cannot be empty")
    if any(value < 0 for value in tolerances):
        raise ValueError("tolerances must be non-negative")

    reports: list[ToleranceReport] = []
    for tolerance in tolerances:
        rows = build_dataset(ticks, horizon_seconds=horizon_seconds, tolerance=tolerance)
        train, test = temporal_split(rows, train_ratio)
        baseline = evaluate_baselines(train, test)
        labeled = [row for row in rows if row.label is not None]
        flat = sum(row.label == "FLAT" for row in labeled)

        non_overlapping = sample_non_overlapping(rows, horizon_seconds)
        non_train, non_test = temporal_split(non_overlapping, train_ratio)
        non_overlap_accuracy = None
        non_overlap_shape = None
        if non_train and non_test and any(row.label is not None for row in non_train):
            non_overlap_accuracy = evaluate_baselines(non_train, non_test).persistence.accuracy
            non_overlap_shape = _shape(non_test, non_train)

        reports.append(
            ToleranceReport(
                tolerance=tolerance,
                rows=len(rows),
                train_rows=len(train),
                test_rows=len(test),
                flat_ratio=flat / len(labeled) if labeled else 0.0,
                persistence_accuracy=baseline.persistence.accuracy,
                non_overlapping_persistence_accuracy=non_overlap_accuracy,
                non_overlapping_rows=len(non_overlapping),
                persistence_shape=_shape(test, train),
                non_overlapping_persistence_shape=non_overlap_shape,
            )
        )
    return tuple(reports)
