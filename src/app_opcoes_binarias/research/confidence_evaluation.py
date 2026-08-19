"""Confidence-binned out-of-sample evaluation for the Softmax model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .baselines import accuracy
from .dataset import ResearchRow
from .model_evaluation import _features
from .models import SoftmaxClassifier

CONFIDENCE_BINS = (
    (0.0, 0.50, "<0.50"),
    (0.50, 0.55, "0.50-0.55"),
    (0.55, 0.60, "0.55-0.60"),
    (0.60, 0.70, "0.60-0.70"),
    (0.70, 1.01, ">=0.70"),
)


@dataclass(frozen=True)
class ConfidenceBin:
    name: str
    rows: int
    accuracy: float
    correct: int


@dataclass(frozen=True)
class ConfidenceReport:
    usable_rows: int
    skipped_rows: int
    mean_confidence: float
    bins: tuple[ConfidenceBin, ...]
    prediction_distribution: dict[str, int]


def evaluate_softmax_confidence(
    train: list[ResearchRow], test: list[ResearchRow]
) -> ConfidenceReport:
    """Measure out-of-sample accuracy as a function of Softmax confidence."""
    classifier = SoftmaxClassifier.fit(train)
    bucket_truth: dict[str, list[str]] = {name: [] for _, _, name in CONFIDENCE_BINS}
    bucket_pred: dict[str, list[str]] = {name: [] for _, _, name in CONFIDENCE_BINS}
    confidences: list[float] = []
    predictions: list[str] = []
    skipped = 0

    for row in test:
        if row.label is None or _features(row) is None:
            skipped += 1
            continue
        probabilities = classifier.probabilities(row)
        if not probabilities:
            skipped += 1
            continue
        prediction = max(probabilities, key=probabilities.get)
        confidence = probabilities[prediction]
        confidences.append(confidence)
        predictions.append(prediction)
        for lower, upper, name in CONFIDENCE_BINS:
            if lower <= confidence < upper:
                bucket_truth[name].append(row.label)
                bucket_pred[name].append(prediction)
                break

    if not confidences:
        raise ValueError("test data contains no complete probability rows")

    bins = tuple(
        ConfidenceBin(
            name=name,
            rows=len(bucket_truth[name]),
            accuracy=accuracy(bucket_truth[name], bucket_pred[name]).accuracy
            if bucket_truth[name]
            else 0.0,
            correct=accuracy(bucket_truth[name], bucket_pred[name]).correct
            if bucket_truth[name]
            else 0,
        )
        for _, _, name in CONFIDENCE_BINS
    )
    return ConfidenceReport(
        usable_rows=len(confidences),
        skipped_rows=skipped,
        mean_confidence=sum(confidences) / len(confidences),
        bins=bins,
        prediction_distribution=dict(Counter(predictions)),
    )
