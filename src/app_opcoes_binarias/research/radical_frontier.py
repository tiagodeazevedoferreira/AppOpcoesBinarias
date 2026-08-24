"""Precision-frontier diagnostics for the radical selective research model.

This module does not change promotion logic. It answers a more important
research question first: *where is the conditional edge, if any?*

Instead of asking the existing model to jump directly to a 99% gate, the
frontier measures out-of-sample precision and coverage produced by training
state purity thresholds. This prevents a zero-decision result from hiding
whether the representation contains a useful high-precision subpopulation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .dataset import ResearchRow
from .radical_edge import _StateIndex, _keys, _wilson_lower, fit_radical_index


@dataclass(frozen=True)
class FrontierPoint:
    view: str
    training_lower_threshold: float
    min_evidence: int
    min_views: int
    accuracy: float
    correct: int
    decisions: int
    decision_rate: float
    lower_bound: float
    max_training_evidence: int


@dataclass(frozen=True)
class FrontierReport:
    points: tuple[FrontierPoint, ...]
    best_accuracy: FrontierPoint | None
    best_lower_bound: FrontierPoint | None
    best_coverage_at_99: FrontierPoint | None


def _evaluate_selection(
    train: list[ResearchRow],
    test: list[ResearchRow],
    *,
    threshold: float,
    min_evidence: int,
    min_views: int,
) -> list[FrontierPoint]:
    train = sorted(train, key=lambda row: row.epoch)
    test = sorted(test, key=lambda row: row.epoch)
    prices = [row.quote for row in train + test]
    train_count = len(train)
    index: _StateIndex = fit_radical_index(train, stride=60)

    view_names = ("state", "trend", "motif")
    stats: dict[str, list[int]] = {
        view: [0, 0, 0] for view in view_names
    }
    combined = [0, 0, 0, 0]
    max_evidence = {view: 0 for view in view_names}

    for offset, row in enumerate(test):
        if row.label not in {"RISE", "FALL"}:
            continue
        keys = _keys(prices, train_count + offset)
        candidates: list[tuple[str, str, int, float]] = []
        for view in view_names:
            key = keys.get(view)
            if key is None:
                continue
            evidence = index.evidence(view, key)
            if evidence is None or evidence.total < min_evidence:
                continue
            direction, correct = evidence.best()
            if direction is None:
                continue
            lower = _wilson_lower(correct, evidence.total)
            max_evidence[view] = max(max_evidence[view], evidence.total)
            if lower >= threshold:
                candidates.append((view, direction, evidence.total, lower))

        selected = candidates if min_views == 1 else [c for c in candidates if sum(c[1] == x[1] for x in candidates) >= min_views]
        if not selected:
            continue

        directions = [candidate[1] for candidate in selected]
        if len(set(directions)) != 1:
            continue
        direction = directions[0]
        combined[0] += 1
        combined[1] += int(direction == row.label)
        combined[2] = max(combined[2], max(candidate[2] for candidate in selected))
        combined[3] += min(candidate[3] for candidate in selected)
        for view, candidate_direction, evidence_total, lower in selected:
            if candidate_direction != direction:
                continue
            stats[view][0] += 1
            stats[view][1] += int(candidate_direction == row.label)
            stats[view][2] = max(stats[view][2], evidence_total)

    total_rows = sum(row.label in {"RISE", "FALL"} for row in test)
    points: list[FrontierPoint] = []
    for view in view_names:
        decisions, correct, _ = stats[view]
        points.append(
            FrontierPoint(
                view=view,
                training_lower_threshold=threshold,
                min_evidence=min_evidence,
                min_views=1,
                accuracy=correct / decisions if decisions else 0.0,
                correct=correct,
                decisions=decisions,
                decision_rate=decisions / total_rows if total_rows else 0.0,
                lower_bound=_wilson_lower(correct, decisions),
                max_training_evidence=max_evidence[view],
            )
        )

    decisions, correct, max_ev, lower_sum = combined
    points.append(
        FrontierPoint(
            view="consensus",
            training_lower_threshold=threshold,
            min_evidence=min_evidence,
            min_views=min_views,
            accuracy=correct / decisions if decisions else 0.0,
            correct=correct,
            decisions=decisions,
            decision_rate=decisions / total_rows if total_rows else 0.0,
            lower_bound=_wilson_lower(correct, decisions),
            max_training_evidence=max_ev,
        )
    )
    return points


def evaluate_frontier(
    train: Iterable[ResearchRow],
    test: Iterable[ResearchRow],
    *,
    thresholds: Iterable[float] = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.985, 0.99),
    evidence_levels: Iterable[int] = (20, 50, 100),
    min_views_levels: Iterable[int] = (1, 2),
) -> FrontierReport:
    train_rows = list(train)
    test_rows = list(test)
    points: list[FrontierPoint] = []
    for threshold in thresholds:
        for min_evidence in evidence_levels:
            for min_views in min_views_levels:
                points.extend(
                    _evaluate_selection(
                        train_rows,
                        test_rows,
                        threshold=threshold,
                        min_evidence=min_evidence,
                        min_views=min_views,
                    )
                )

    eligible = [point for point in points if point.decisions > 0]
    best_accuracy = max(eligible, key=lambda point: (point.accuracy, point.lower_bound, point.decisions), default=None)
    best_lower_bound = max(eligible, key=lambda point: (point.lower_bound, point.accuracy, point.decisions), default=None)
    at_99 = [point for point in eligible if point.accuracy >= 0.99]
    best_coverage_at_99 = max(at_99, key=lambda point: (point.decision_rate, point.lower_bound), default=None)
    return FrontierReport(tuple(points), best_accuracy, best_lower_bound, best_coverage_at_99)


def as_report_dict(report: FrontierReport) -> dict:
    return {
        "points": [asdict(point) for point in report.points],
        "best_accuracy": asdict(report.best_accuracy) if report.best_accuracy else None,
        "best_lower_bound": asdict(report.best_lower_bound) if report.best_lower_bound else None,
        "best_coverage_at_99": asdict(report.best_coverage_at_99) if report.best_coverage_at_99 else None,
    }
