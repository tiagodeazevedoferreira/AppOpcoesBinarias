"""Leakage-safe baseline evaluation for directional research."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .baselines import ClassificationMetrics, accuracy, majority_class, persistence_prediction
from .dataset import ResearchRow


@dataclass(frozen=True)
class BaselineReport:
    majority: ClassificationMetrics
    persistence: ClassificationMetrics
    train_distribution: dict[str, int]
    test_distribution: dict[str, int]


def evaluate_baselines(train: list[ResearchRow], test: list[ResearchRow]) -> BaselineReport:
    """Evaluate simple baselines using only chronological train/test information."""
    train_labeled = [row.label for row in train if row.label is not None]
    test_labeled = [row for row in test if row.label is not None]
    if not train_labeled:
        raise ValueError("train must contain labeled rows")

    majority = majority_class(train_labeled)
    majority_metrics = accuracy(
        [row.label for row in test_labeled],
        [majority for _ in test_labeled],
    )

    persistence_truth: list[str] = []
    persistence_pred: list[str | None] = []
    previous: str | None = train_labeled[-1]
    for row in test_labeled:
        persistence_truth.append(row.label)  # type: ignore[arg-type]
        persistence_pred.append(persistence_prediction(previous))
        previous = row.label

    return BaselineReport(
        majority=majority_metrics,
        persistence=accuracy(persistence_truth, persistence_pred),
        train_distribution=dict(Counter(train_labeled)),
        test_distribution=dict(Counter(row.label for row in test_labeled)),
    )


def sample_non_overlapping(rows: list[ResearchRow], horizon_seconds: int) -> list[ResearchRow]:
    """Select decision rows whose observation times are at least one horizon apart."""
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    ordered = sorted(rows, key=lambda row: row.epoch)
    selected: list[ResearchRow] = []
    next_allowed: int | None = None
    for row in ordered:
        if row.label is None:
            continue
        if next_allowed is None or row.epoch >= next_allowed:
            selected.append(row)
            next_allowed = row.epoch + horizon_seconds
    return selected
