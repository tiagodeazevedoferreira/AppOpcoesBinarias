"""Deterministic feature-based models for leakage-safe directional research."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .dataset import ResearchRow

FEATURE_NAMES = (
    "return_1",
    "momentum_2",
    "volatility_5",
    "ema_distance_10",
    "directional_consistency_5",
)


@dataclass(frozen=True)
class FeatureStats:
    means: tuple[float, ...]
    scales: tuple[float, ...]


def _features(row: ResearchRow) -> tuple[float, ...] | None:
    values = tuple(getattr(row, name) for name in FEATURE_NAMES)
    if any(
        value is None
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        return None
    return tuple(float(value) for value in values)


def _fit_stats(rows: list[ResearchRow]) -> FeatureStats:
    vectors = [_features(row) for row in rows]
    usable = [vector for vector in vectors if vector is not None]
    if not usable:
        raise ValueError("training data contains no complete feature rows")
    means = tuple(
        sum(vector[i] for vector in usable) / len(usable)
        for i in range(len(FEATURE_NAMES))
    )
    scales = tuple(
        math.sqrt(sum((vector[i] - means[i]) ** 2 for vector in usable) / len(usable)) or 1.0
        for i in range(len(FEATURE_NAMES))
    )
    return FeatureStats(means=means, scales=scales)


def _normalize(vector: tuple[float, ...], stats: FeatureStats) -> tuple[float, ...]:
    return tuple(
        (value - mean) / scale
        for value, mean, scale in zip(vector, stats.means, stats.scales)
    )


@dataclass(frozen=True)
class NearestCentroidClassifier:
    """Classify using standardized distance to train-only class centroids."""

    stats: FeatureStats
    centroids: dict[str, tuple[float, ...]]
    default_class: str

    @classmethod
    def fit(cls, rows: list[ResearchRow]) -> NearestCentroidClassifier:
        labeled = [
            row for row in rows if row.label is not None and _features(row) is not None
        ]
        if not labeled:
            raise ValueError("training data contains no complete labeled feature rows")
        stats = _fit_stats(labeled)
        grouped: dict[str, list[tuple[float, ...]]] = {}
        for row in labeled:
            vector = _features(row)
            assert vector is not None
            grouped.setdefault(row.label, []).append(_normalize(vector, stats))
        centroids = {
            label: tuple(
                sum(vector[i] for vector in vectors) / len(vectors)
                for i in range(len(FEATURE_NAMES))
            )
            for label, vectors in grouped.items()
        }
        default_class = Counter(row.label for row in labeled).most_common(1)[0][0]
        return cls(stats=stats, centroids=centroids, default_class=default_class)

    def predict(self, row: ResearchRow) -> str | None:
        vector = _features(row)
        if vector is None:
            return None
        normalized = _normalize(vector, self.stats)
        return min(
            self.centroids,
            key=lambda label: sum(
                (a - b) ** 2 for a, b in zip(normalized, self.centroids[label])
            ),
        )


@dataclass(frozen=True)
class SoftmaxClassifier:
    """Multiclass linear classifier trained only on the supplied training rows."""

    stats: FeatureStats
    labels: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    learning_rate: float
    epochs: int

    @classmethod
    def fit(
        cls,
        rows: list[ResearchRow],
        *,
        learning_rate: float = 0.05,
        epochs: int = 300,
    ) -> SoftmaxClassifier:
        if learning_rate <= 0 or epochs <= 0:
            raise ValueError("learning_rate and epochs must be positive")
        labeled = [row for row in rows if row.label is not None and _features(row) is not None]
        if not labeled:
            raise ValueError("training data contains no complete labeled feature rows")
        stats = _fit_stats(labeled)
        vectors = [_normalize(_features(row), stats) for row in labeled]
        labels = tuple(sorted({row.label for row in labeled}))
        index = {label: i for i, label in enumerate(labels)}
        width = len(FEATURE_NAMES)
        weights = [[0.0] * width for _ in labels]
        bias = [0.0] * len(labels)
        for _ in range(epochs):
            grad_w = [[0.0] * width for _ in labels]
            grad_b = [0.0] * len(labels)
            for vector, row in zip(vectors, labeled):
                logits = [
                    sum(w * x for w, x in zip(weights[k], vector)) + bias[k]
                    for k in range(len(labels))
                ]
                maximum = max(logits)
                exp_values = [math.exp(min(50.0, value - maximum)) for value in logits]
                total = sum(exp_values)
                probabilities = [value / total for value in exp_values]
                target = index[row.label]
                for k, probability in enumerate(probabilities):
                    error = probability - (1.0 if k == target else 0.0)
                    for j, value in enumerate(vector):
                        grad_w[k][j] += error * value
                    grad_b[k] += error
            scale = 1.0 / len(vectors)
            for k in range(len(labels)):
                for j in range(width):
                    weights[k][j] -= learning_rate * grad_w[k][j] * scale
                bias[k] -= learning_rate * grad_b[k] * scale
        return cls(
            stats=stats,
            labels=labels,
            weights=tuple(tuple(row) for row in weights),
            bias=tuple(bias),
            learning_rate=learning_rate,
            epochs=epochs,
        )

    def probabilities(self, row: ResearchRow) -> dict[str, float] | None:
        vector = _features(row)
        if vector is None:
            return None
        normalized = _normalize(vector, self.stats)
        logits = [
            sum(w * x for w, x in zip(weights, normalized)) + bias
            for weights, bias in zip(self.weights, self.bias)
        ]
        maximum = max(logits)
        exp_values = [math.exp(min(50.0, value - maximum)) for value in logits]
        total = sum(exp_values)
        return {label: value / total for label, value in zip(self.labels, exp_values)}

    def predict(self, row: ResearchRow) -> str | None:
        probabilities = self.probabilities(row)
        if probabilities is None:
            return None
        return max(probabilities, key=probabilities.get)
