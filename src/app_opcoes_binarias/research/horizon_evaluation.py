from __future__ import annotations

from dataclasses import dataclass

from .baselines import ClassificationMetrics
from .dataset import ResearchRow, temporal_split
from .evaluation import evaluate_baselines, sample_non_overlapping


@dataclass(frozen=True)
class HorizonReport:
    horizon_seconds: int
    rows: int
    train_rows: int
    test_rows: int
    majority: ClassificationMetrics
    persistence: ClassificationMetrics
    non_overlapping_persistence: ClassificationMetrics


def evaluate_horizons(
    rows_by_horizon: dict[int, list[ResearchRow]], *, train_ratio: float = 0.7
) -> tuple[HorizonReport, ...]:
    """Compare simple baselines across prediction horizons without cross-horizon leakage."""
    reports: list[HorizonReport] = []
    for horizon in sorted(rows_by_horizon):
        rows = rows_by_horizon[horizon]
        train, test = temporal_split(rows, train_ratio)
        if not train or not test:
            continue
        baseline = evaluate_baselines(train, test)
        non_overlapping = sample_non_overlapping(rows, horizon)
        _, non_overlap_test = temporal_split(non_overlapping, train_ratio)
        non_overlap_baseline = evaluate_baselines(
            non_overlapping[: len(non_overlapping) - len(non_overlap_test)],
            non_overlap_test,
        )
        reports.append(
            HorizonReport(
                horizon_seconds=horizon,
                rows=len(rows),
                train_rows=len(train),
                test_rows=len(test),
                majority=baseline.majority,
                persistence=baseline.persistence,
                non_overlapping_persistence=non_overlap_baseline.persistence,
            )
        )
    return tuple(reports)
