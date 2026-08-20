from __future__ import annotations

from dataclasses import dataclass

from .dataset import build_dataset, temporal_split
from .evaluation import evaluate_baselines, sample_non_overlapping


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
        if non_train and non_test and any(row.label is not None for row in non_train):
            non_overlap_accuracy = evaluate_baselines(non_train, non_test).persistence.accuracy

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
            )
        )
    return tuple(reports)
