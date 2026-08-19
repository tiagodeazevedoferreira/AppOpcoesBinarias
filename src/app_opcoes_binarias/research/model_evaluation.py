"""Out-of-sample evaluation helpers for research models."""

from __future__ import annotations

from dataclasses import dataclass

from .baselines import ClassificationMetrics, accuracy
from .dataset import ResearchRow
from .models import NearestCentroidClassifier, SoftmaxClassifier, _features


@dataclass(frozen=True)
class ModelReport:
    model: ClassificationMetrics
    usable_test_rows: int
    skipped_test_rows: int


def _evaluate_predictions(train: list[ResearchRow], test: list[ResearchRow], predictor) -> ModelReport:
    truth: list[str] = []
    predictions: list[str] = []
    skipped = 0
    for row in test:
        if row.label is None or _features(row) is None:
            skipped += 1
            continue
        prediction = predictor(row)
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


def evaluate_nearest_centroid(
    train: list[ResearchRow], test: list[ResearchRow]
) -> ModelReport:
    """Fit on train only and evaluate complete-feature test rows."""
    classifier = NearestCentroidClassifier.fit(train)
    return _evaluate_predictions(train, test, classifier.predict)


def evaluate_softmax(
    train: list[ResearchRow],
    test: list[ResearchRow],
    *,
    learning_rate: float = 0.05,
    epochs: int = 300,
) -> ModelReport:
    """Fit a train-only linear softmax classifier and evaluate it out of sample."""
    classifier = SoftmaxClassifier.fit(train, learning_rate=learning_rate, epochs=epochs)
    return _evaluate_predictions(train, test, classifier.predict)
