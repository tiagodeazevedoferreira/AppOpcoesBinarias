"""Simple, leakage-safe baselines for directional research."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    total: int
    correct: int


def majority_class(train_labels: Sequence[str]) -> str:
    """Return the most frequent label using only the supplied training set."""
    labels = [label for label in train_labels if label]
    if not labels:
        raise ValueError("train_labels cannot be empty")
    counts = Counter(labels)
    # Stable tie-breaking keeps experiments reproducible.
    return min(counts, key=lambda label: (-counts[label], label))


def persistence_prediction(previous_direction: str | None) -> str | None:
    """Predict the next direction as the most recently observed direction."""
    return previous_direction


def accuracy(y_true: Sequence[str], y_pred: Sequence[str | None]) -> ClassificationMetrics:
    """Calculate exact-match accuracy, ignoring predictions that are unavailable."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    evaluated = [(truth, pred) for truth, pred in zip(y_true, y_pred) if pred is not None]
    if not evaluated:
        return ClassificationMetrics(accuracy=0.0, total=0, correct=0)
    correct = sum(truth == pred for truth, pred in evaluated)
    return ClassificationMetrics(
        accuracy=correct / len(evaluated),
        total=len(evaluated),
        correct=correct,
    )
