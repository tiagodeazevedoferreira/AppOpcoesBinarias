"""Out-of-sample evaluation helpers for research models."""

from __future__ import annotations

from dataclasses import dataclass

from .baselines import ClassificationMetrics, accuracy
from .dataset import ResearchRow
from .models import NearestCentroidClassifier, _features


@dataclass(frozen=True)
class ModelReport:
    model: ClassificationMetrics
    usable_test_rows: int
    skipped_test_rows: int


def evaluate_nearest_centroid(
    train: list[ResearchRow], test: list[ResearchRow]
) -> ModelReport:
    """Fit on train only and evaluate complete-feature test rows."""
    classifier = NearestCentroidClassifier.fit(train)
    truth: list[str] = []
    predictions: list[str] = []
    skipped = 0
    for row in test:
        if row.label is None or _features(row) is None:
            skipped += 1
            continue
        prediction = classifier.predict(row)
        if prediction is None:
            skipped += 1
            continue
        truth.append(row.label)
        predictions.append(prediction)
    if not truth:
        raise ValueError("test data contains no complete labeled feature rows")
    return ModelReport(
        model=accuracy(truth, predictions),
        usable_test_rows=len(truth),
        skipped_test_rows=skipped,
    )
