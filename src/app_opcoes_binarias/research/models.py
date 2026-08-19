"""Deterministic feature-based models for leakage-safe directional research."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

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
    if any(value is None or not math.isfinite(value) for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[arg-type]


def _fit_stats(rows: list[ResearchRow]) -> FeatureStats:
    vectors = [_features(row) for row in rows]
    usable = [vector for vector in vectors if vector is not None]
    if not usable:
        raise ValueError("training data contains no complete feature rows")
    means = tuple(sum(vector[i] for vector in usable) / len(usable) for i in range(len(FEATURE_NAMES)))
    scales = tuple(
        math.sqrt(sum((vector[i] - means[i]) ** 2 for vector in usable) / len(usable)) or 1.0
        for i in range(len(FEATURE_NAMES))
    )
    return FeatureStats(means=means, scales=scales)


def _normalize(vector: tuple[float, ...], stats: FeatureStats) -> tuple[float, ...]:
    return tuple((value - mean) / scale for value, mean, scale in zip(vector, stats.means, stats.scales))


@dataclass(frozen=True)
class NearestCentroidClassifier:
    """Classify using standardized distance to train-only class centroids."""

    stats: FeatureStats
    centroids: dict[str, tuple[float, ...]]
    default_class: str

    @classmethod
    def fit(cls, rows: list[ResearchRow]) -> "NearestCentroidClassifier":
        labeled = [row for row in rows if row.label is not None and _features(row) is not None]
        if not labeled:
            raise ValueError("training data contains no complete labeled feature rows")
        stats = _fit_stats(labeled)
        grouped: dict[str, list[tuple[float, ...]]] = {}
        for row in labeled:
            vector = _features(row)
            assert vector is not None
            grouped.setdefault(row.label, []).append(_normalize(vector, stats))
        centroids = {
            label: tuple(sum(vector[i] for vector in vectors) / len(vectors) for i in range(len(FEATURE_NAMES)))
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
            key=lambda label: sum((a - b) ** 2 for a, b in zip(normalized, self.centroids[label])),
        )
