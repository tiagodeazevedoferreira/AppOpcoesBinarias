"""Selective, leakage-safe radical-edge research.

This module deliberately changes the objective from "predict every tick" to
"act only when historical evidence makes the next outcome unusually
predictable".  It combines several past-only state representations and uses a
one-sided Wilson lower confidence bound before allowing a directional
prediction.

The model never reads a future quote while constructing a state.  Labels are
used only to build the training lookup tables, and test labels are touched
only after a decision has been made.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .dataset import ResearchRow


@dataclass(frozen=True)
class RadicalPrediction:
    direction: str | None
    confidence: float
    lower_bound: float
    evidence: int
    source: str


@dataclass(frozen=True)
class RadicalMetrics:
    accuracy: float
    correct: int
    total_decisions: int
    decision_rate: float
    total_rows: int
    no_bet_decisions: int
    target_accuracy: float
    coverage_at_target: float
    mean_lower_bound: float
    max_evidence: int


@dataclass(frozen=True)
class RadicalFold:
    train_rows: int
    test_rows: int
    metrics: RadicalMetrics


@dataclass(frozen=True)
class RadicalWalkForward:
    folds: tuple[RadicalFold, ...]
    aggregate: RadicalMetrics


@dataclass
class _Evidence:
    rise: int = 0
    fall: int = 0

    @property
    def total(self) -> int:
        return self.rise + self.fall

    def add(self, label: str) -> None:
        if label == "RISE":
            self.rise += 1
        elif label == "FALL":
            self.fall += 1

    def best(self) -> tuple[str | None, int]:
        if self.total == 0:
            return None, 0
        if self.rise >= self.fall:
            return "RISE", self.rise
        return "FALL", self.fall


class _StateIndex:
    def __init__(self) -> None:
        self._tables: dict[str, dict[tuple[int, ...], _Evidence]] = {
            "state": defaultdict(_Evidence),
            "motif": defaultdict(_Evidence),
            "trend": defaultdict(_Evidence),
        }

    def add(self, keys: dict[str, tuple[int, ...]], label: str | None) -> None:
        if label not in {"RISE", "FALL"}:
            return
        for name, key in keys.items():
            self._tables[name][key].add(label)

    def evidence(self, name: str, key: tuple[int, ...]) -> _Evidence | None:
        return self._tables[name].get(key)


def _sign(value: float, epsilon: float = 0.0) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _quantile_bucket(value: float, cuts: Sequence[float]) -> int:
    for index, cut in enumerate(cuts):
        if value <= cut:
            return index
    return len(cuts)


def _history(prices: Sequence[float], index: int, lag: int) -> float | None:
    if index < lag:
        return None
    previous = prices[index - lag]
    current = prices[index]
    if previous == 0:
        return None
    return current / previous - 1.0


def _run_length(prices: Sequence[float], index: int, maximum: int = 60) -> int:
    if index < 1:
        return 0
    last = _sign(prices[index] - prices[index - 1])
    if last == 0:
        return 0
    length = 0
    for position in range(index, max(0, index - maximum), -1):
        if position < 1:
            break
        direction = _sign(prices[position] - prices[position - 1])
        if direction != last:
            break
        length += 1
    return length


def _efficiency(prices: Sequence[float], index: int, window: int = 60) -> float | None:
    if index < window:
        return None
    start = index - window
    displacement = abs(prices[index] - prices[start])
    path = sum(abs(prices[p] - prices[p - 1]) for p in range(start + 1, index + 1))
    if path == 0:
        return 0.0
    return displacement / path


def _volatility(prices: Sequence[float], index: int, window: int = 30) -> float | None:
    if index < window:
        return None
    values = []
    for p in range(index - window + 1, index + 1):
        if p < 1 or prices[p - 1] == 0:
            continue
        values.append(prices[p] / prices[p - 1] - 1.0)
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def _motif_code(prices: Sequence[float], index: int, window: int = 12) -> tuple[int, ...] | None:
    if index < window:
        return None
    return tuple(_sign(prices[p] - prices[p - 1]) + 1 for p in range(index - window + 1, index + 1))


def _keys(prices: Sequence[float], index: int) -> dict[str, tuple[int, ...]]:
    r15 = _history(prices, index, 15)
    r30 = _history(prices, index, 30)
    r60 = _history(prices, index, 60)
    if r15 is None or r30 is None or r60 is None:
        return {}

    signs = (_sign(r15), _sign(r30), _sign(r60))
    consistency_values = []
    for window in (15, 30, 60):
        if index < window:
            return {}
        non_flat = [
            _sign(prices[p] - prices[p - 1])
            for p in range(index - window + 1, index + 1)
            if prices[p] != prices[p - 1]
        ]
        if not non_flat:
            consistency_values.append(0.0)
        else:
            consistency_values.append(max(sum(x > 0 for x in non_flat), sum(x < 0 for x in non_flat)) / len(non_flat))

    efficiency = _efficiency(prices, index, 60)
    volatility = _volatility(prices, index, 30)
    run = _run_length(prices, index)
    if efficiency is None or volatility is None:
        return {}

    agreement = sum(1 for value in signs if value == signs[-1])
    state = (
        *signs,
        _quantile_bucket(consistency_values[2], (0.55, 0.65, 0.75, 0.85, 0.95)),
        _quantile_bucket(efficiency, (0.10, 0.20, 0.35, 0.50, 0.70, 0.85)),
        min(run // 3, 8),
        agreement,
        _quantile_bucket(math.log10(max(volatility, 1e-12)), (-5.0, -4.5, -4.0, -3.5, -3.0, -2.5)),
    )
    trend = (
        signs[-1],
        _quantile_bucket(consistency_values[2], (0.60, 0.70, 0.80, 0.90, 0.96)),
        _quantile_bucket(efficiency, (0.15, 0.30, 0.50, 0.70, 0.85)),
        min(run // 5, 6),
        agreement,
    )
    keys: dict[str, tuple[int, ...]] = {"state": state, "trend": trend}
    motif = _motif_code(prices, index, 12)
    if motif is not None:
        keys["motif"] = motif
    return keys


def _wilson_lower(correct: int, total: int, z: float = 2.326347874) -> float:
    """One-sided Wilson lower bound (99% confidence by default)."""
    if total <= 0:
        return 0.0
    p = correct / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    spread = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return max(0.0, (center - spread) / denominator)


def fit_radical_index(rows: Sequence[ResearchRow]) -> _StateIndex:
    """Fit lookup evidence using only past-state keys and known training labels."""
    prices = [row.quote for row in rows]
    index = _StateIndex()
    for position, row in enumerate(rows):
        if row.label is None:
            continue
        keys = _keys(prices, position)
        index.add(keys, row.label)
    return index


def predict_radical(
    index: _StateIndex,
    prices: Sequence[float],
    position: int,
    *,
    target_accuracy: float = 0.99,
    min_evidence: int = 100,
    min_agreement: int = 2,
) -> RadicalPrediction:
    """Predict only when independent historical state views support the same edge."""
    keys = _keys(prices, position)
    if not keys:
        return RadicalPrediction(None, 0.0, 0.0, 0, "insufficient_history")

    candidates: list[tuple[str, _Evidence, float]] = []
    for name, key in keys.items():
        evidence = index.evidence(name, key)
        if evidence is None or evidence.total < min_evidence:
            continue
        direction, correct = evidence.best()
        if direction is None:
            continue
        lower = _wilson_lower(correct, evidence.total)
        confidence = correct / evidence.total
        candidates.append((name, evidence, lower))

    if not candidates:
        return RadicalPrediction(None, 0.0, 0.0, 0, "insufficient_evidence")

    strong = [candidate for candidate in candidates if candidate[2] >= target_accuracy]
    if len(strong) < min_agreement:
        return RadicalPrediction(None, 0.0, max(candidate[2] for candidate in candidates), max(candidate[1].total for candidate in candidates), "no_consensus")

    directions = [candidate[1].best()[0] for candidate in strong]
    if len(set(directions)) != 1:
        return RadicalPrediction(None, 0.0, 0.0, 0, "conflicting_evidence")

    strong.sort(key=lambda item: (item[2], item[1].total), reverse=True)
    direction, evidence_total = strong[0][1].best()
    weighted_confidence = sum(candidate[1].best()[1] for candidate in strong) / sum(candidate[1].total for candidate in strong)
    lower = min(candidate[2] for candidate in strong)
    return RadicalPrediction(direction, weighted_confidence, lower, evidence_total, "+".join(candidate[0] for candidate in strong))


def evaluate_radical(
    train: Sequence[ResearchRow],
    test: Sequence[ResearchRow],
    *,
    target_accuracy: float = 0.99,
    min_evidence: int = 100,
) -> RadicalMetrics:
    """Evaluate the selective model strictly out of sample."""
    train_rows = sorted(train, key=lambda row: row.epoch)
    test_rows = sorted(test, key=lambda row: row.epoch)
    combined = list(train_rows) + list(test_rows)
    train_count = len(train_rows)
    prices = [row.quote for row in combined]
    index = fit_radical_index(train_rows)

    correct = 0
    decisions = 0
    lower_sum = 0.0
    max_evidence = 0
    for offset, row in enumerate(test_rows):
        position = train_count + offset
        prediction = predict_radical(index, prices, position, target_accuracy=target_accuracy, min_evidence=min_evidence)
        if prediction.direction is None or row.label not in {"RISE", "FALL"}:
            continue
        decisions += 1
        correct += int(prediction.direction == row.label)
        lower_sum += prediction.lower_bound
        max_evidence = max(max_evidence, prediction.evidence)

    total_rows = sum(1 for row in test_rows if row.label in {"RISE", "FALL"})
    accuracy = correct / decisions if decisions else 0.0
    return RadicalMetrics(
        accuracy=accuracy,
        correct=correct,
        total_decisions=decisions,
        decision_rate=decisions / total_rows if total_rows else 0.0,
        total_rows=total_rows,
        no_bet_decisions=max(0, total_rows - decisions),
        target_accuracy=target_accuracy,
        coverage_at_target=decisions / total_rows if total_rows else 0.0,
        mean_lower_bound=lower_sum / decisions if decisions else 0.0,
        max_evidence=max_evidence,
    )


def evaluate_radical_walk_forward(
    rows: Sequence[ResearchRow],
    *,
    folds: int = 5,
    target_accuracy: float = 0.99,
    min_evidence: int = 100,
) -> RadicalWalkForward:
    """Walk forward without ever training on future labels."""
    ordered = sorted(rows, key=lambda row: row.epoch)
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(ordered) < folds + 1:
        raise ValueError("not enough rows for walk-forward evaluation")

    fold_size = len(ordered) // (folds + 1)
    fold_reports: list[RadicalFold] = []
    correct = decisions = total_rows = 0
    lower_sum = 0.0
    max_evidence = 0
    for fold in range(folds):
        train_end = fold_size * (fold + 1)
        test_end = fold_size * (fold + 2) if fold < folds - 1 else len(ordered)
        train = ordered[:train_end]
        test = ordered[train_end:test_end]
        metrics = evaluate_radical(train, test, target_accuracy=target_accuracy, min_evidence=min_evidence)
        fold_reports.append(RadicalFold(len(train), len(test), metrics))
        correct += metrics.correct
        decisions += metrics.total_decisions
        total_rows += metrics.total_rows
        lower_sum += metrics.mean_lower_bound * metrics.total_decisions
        max_evidence = max(max_evidence, metrics.max_evidence)

    aggregate = RadicalMetrics(
        accuracy=correct / decisions if decisions else 0.0,
        correct=correct,
        total_decisions=decisions,
        decision_rate=decisions / total_rows if total_rows else 0.0,
        total_rows=total_rows,
        no_bet_decisions=max(0, total_rows - decisions),
        target_accuracy=target_accuracy,
        coverage_at_target=decisions / total_rows if total_rows else 0.0,
        mean_lower_bound=lower_sum / decisions if decisions else 0.0,
        max_evidence=max_evidence,
    )
    return RadicalWalkForward(tuple(fold_reports), aggregate)
