from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Iterable

from .dataset import ResearchRow

DIRECTIONS = ("RISE", "FALL")
DEFAULT_LOOKBACKS = (15, 30, 60, 120, 300)


@dataclass(frozen=True)
class RadicalSignal:
    epoch: int
    direction: str | None
    score: int
    agreement: int
    matured_direction: str | None
    matured_run: int
    momentum_signs: tuple[int, ...]
    confidence: float
    action: str
    reason: str


@dataclass(frozen=True)
class RadicalPrecisionReport:
    target_precision: float
    calibration_precision: float
    test_precision: float
    calibration_decisions: int
    test_decisions: int
    calibration_correct: int
    test_correct: int
    calibration_error: int
    test_error: int
    calibration_coverage: float
    test_coverage: float
    achieved_target: bool
    rule: dict[str, int | float]


def _index(rows: list[ResearchRow]) -> tuple[list[int], list[float]]:
    ordered = sorted(rows, key=lambda row: row.epoch)
    return [row.epoch for row in ordered], [row.quote for row in ordered]


def _price_at_or_before(epochs: list[int], quotes: list[float], target: int) -> float | None:
    index = bisect.bisect_right(epochs, target) - 1
    return quotes[index] if index >= 0 else None


def _momentum(row: ResearchRow, epochs: list[int], quotes: list[float], lookback: int) -> float | None:
    previous = _price_at_or_before(epochs, quotes, row.epoch - lookback)
    if previous is None or previous == 0.0 or row.quote == 0.0:
        return None
    return row.quote / previous - 1.0


def _matured_label(
    row: ResearchRow,
    ordered: list[ResearchRow],
    epochs: list[int],
    horizon: int,
) -> tuple[str | None, int]:
    """Return the newest outcome whose full horizon has already elapsed."""
    cutoff = row.epoch - horizon
    index = bisect.bisect_right(epochs, cutoff) - 1
    if index < 0:
        return None, 0
    direction = ordered[index].label if ordered[index].label in DIRECTIONS else None
    if direction is None:
        return None, 0
    run = 1
    index -= 1
    while index >= 0 and ordered[index].label == direction:
        run += 1
        index -= 1
    return direction, run


def _sign(value: float | None) -> int:
    if value is None or value == 0.0:
        return 0
    return 1 if value > 0 else -1


def _signal_for_row(
    row: ResearchRow,
    ordered: list[ResearchRow],
    epochs: list[int],
    quotes: list[float],
    horizon: int,
    lookbacks: tuple[int, ...],
    min_agreement: int,
    min_run: int,
    min_momentum: float,
) -> RadicalSignal:
    matured, run = _matured_label(row, ordered, epochs, horizon)
    momenta = tuple(_momentum(row, epochs, quotes, lookback) for lookback in lookbacks)
    signs = tuple(_sign(value) for value in momenta)

    votes: list[int] = [sign for sign in signs if sign]
    if matured:
        votes.append(1 if matured == "RISE" else -1)
    positive = sum(vote > 0 for vote in votes)
    negative = sum(vote < 0 for vote in votes)
    score = positive - negative
    agreement = max(positive, negative)
    direction = "RISE" if score > 0 else "FALL" if score < 0 else None

    strongest = max((abs(value) for value in momenta if value is not None), default=0.0)
    matured_ok = matured is not None and run >= min_run
    momentum_ok = strongest >= min_momentum
    if direction is None:
        return RadicalSignal(row.epoch, None, score, agreement, matured, run, signs, 0.0, "NO_BET", "no_direction")
    if not matured_ok:
        return RadicalSignal(row.epoch, None, score, agreement, matured, run, signs, 0.0, "NO_BET", "maturity_gate")
    if not momentum_ok:
        return RadicalSignal(row.epoch, None, score, agreement, matured, run, signs, 0.0, "NO_BET", "momentum_gate")
    if agreement < min_agreement:
        return RadicalSignal(row.epoch, None, score, agreement, matured, run, signs, 0.0, "NO_BET", "agreement_gate")

    confidence = agreement / max(1, len(votes))
    if matured != direction:
        return RadicalSignal(row.epoch, None, score, agreement, matured, run, signs, confidence, "NO_BET", "maturity_disagrees")
    return RadicalSignal(row.epoch, direction, score, agreement, matured, run, signs, confidence, "BET", "all_gates")


def _evaluate(
    context: list[ResearchRow],
    evaluation_rows: list[ResearchRow],
    *,
    horizon: int,
    lookbacks: tuple[int, ...],
    min_agreement: int,
    min_run: int,
    min_momentum: float,
) -> tuple[int, int, int, list[RadicalSignal]]:
    context_ordered = sorted(context, key=lambda row: row.epoch)
    evaluation_ordered = sorted(evaluation_rows, key=lambda row: row.epoch)
    ordered = sorted(context_ordered + evaluation_ordered, key=lambda row: row.epoch)
    epochs, quotes = _index(ordered)
    signals: list[RadicalSignal] = []
    correct = decisions = errors = 0
    for row in evaluation_ordered:
        if row.label not in DIRECTIONS:
            continue
        signal = _signal_for_row(
            row, ordered, epochs, quotes, horizon, lookbacks, min_agreement, min_run, min_momentum
        )
        signals.append(signal)
        if signal.action == "BET":
            decisions += 1
            if signal.direction == row.label:
                correct += 1
            else:
                errors += 1
    return correct, decisions, errors, signals


def _candidate_grid(train: list[ResearchRow], horizon: int) -> Iterable[dict[str, int | float]]:
    ordered = sorted(train, key=lambda row: row.epoch)
    epochs, quotes = _index(ordered)
    magnitudes = []
    for row in ordered:
        if row.label in DIRECTIONS:
            values = [_momentum(row, epochs, quotes, lookback) for lookback in DEFAULT_LOOKBACKS]
            magnitudes.extend(abs(value) for value in values if value is not None and value != 0.0)
    if not magnitudes:
        thresholds = (0.0,)
    else:
        values = sorted(magnitudes)
        thresholds = tuple(
            values[min(len(values) - 1, int(len(values) * q))]
            for q in (0.0, 0.5, 0.7, 0.85, 0.95, 0.99)
        )
    width = len(DEFAULT_LOOKBACKS) + 1
    for min_agreement in range(max(2, width - 2), width + 1):
        for min_run in (1, 2, 3, 5, 10):
            for threshold in thresholds:
                yield {
                    "min_agreement": min_agreement,
                    "min_run": min_run,
                    "min_momentum": threshold,
                }


def search_and_evaluate(
    train: list[ResearchRow],
    test: list[ResearchRow],
    *,
    horizon: int = 60,
    target_precision: float = 0.99,
) -> RadicalPrecisionReport:
    """Search a finite rule family on calibration data, then freeze it for final test."""
    if not 0.0 < target_precision < 1.0:
        raise ValueError("target_precision must be between 0 and 1")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    ordered_train = sorted(train, key=lambda row: row.epoch)
    cut = max(1, int(len(ordered_train) * 0.7))
    fit = ordered_train[:cut]
    calibration = ordered_train[cut:]

    best_rule: dict[str, int | float] | None = None
    best_rank: tuple[int, float, float, int] | None = None
    for rule in _candidate_grid(fit, horizon):
        correct, decisions, _, _ = _evaluate(
            fit,
            fit,
            horizon=horizon,
            lookbacks=DEFAULT_LOOKBACKS,
            **rule,
        )
        if decisions == 0:
            continue
        precision = correct / decisions
        coverage = decisions / max(1, len(fit))
        rank = (1 if precision >= target_precision else 0, precision, coverage, decisions)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_rule = rule

    rule = best_rule or {
        "min_agreement": len(DEFAULT_LOOKBACKS) + 1,
        "min_run": 1,
        "min_momentum": 0.0,
    }

    cal_correct, cal_decisions, cal_errors, _ = _evaluate(
        fit,
        calibration,
        horizon=horizon,
        lookbacks=DEFAULT_LOOKBACKS,
        **rule,
    )
    test_context = fit + calibration
    test_correct, test_decisions, test_errors, _ = _evaluate(
        test_context,
        test,
        horizon=horizon,
        lookbacks=DEFAULT_LOOKBACKS,
        **rule,
    )
    cal_precision = cal_correct / cal_decisions if cal_decisions else 0.0
    test_precision = test_correct / test_decisions if test_decisions else 0.0
    return RadicalPrecisionReport(
        target_precision=target_precision,
        calibration_precision=cal_precision,
        test_precision=test_precision,
        calibration_decisions=cal_decisions,
        test_decisions=test_decisions,
        calibration_correct=cal_correct,
        test_correct=test_correct,
        calibration_error=cal_errors,
        test_error=test_errors,
        calibration_coverage=cal_decisions / len(calibration) if calibration else 0.0,
        test_coverage=test_decisions / len(test) if test else 0.0,
        achieved_target=test_decisions > 0 and test_precision >= target_precision,
        rule=rule,
    )
