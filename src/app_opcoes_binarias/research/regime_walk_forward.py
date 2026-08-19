from __future__ import annotations

from dataclasses import dataclass

from .baselines import ClassificationMetrics
from .dataset import ResearchRow
from .regime_evaluation import evaluate_regime_persistence


@dataclass(frozen=True)
class RegimeFoldReport:
    train_rows: int
    test_rows: int
    regime_persistence: ClassificationMetrics
    directional_decisions: int
    no_bet_decisions: int


@dataclass(frozen=True)
class RegimeWalkForwardReport:
    folds: tuple[RegimeFoldReport, ...]
    regime_persistence: ClassificationMetrics
    directional_decisions: int
    no_bet_decisions: int


def evaluate_regime_walk_forward(
    rows: list[ResearchRow], *, folds: int = 5, window: int = 60
) -> RegimeWalkForwardReport:
    """Evaluate regime-conditioned persistence on expanding chronological folds."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if window <= 0:
        raise ValueError("window must be positive")
    ordered = sorted(rows, key=lambda row: row.epoch)
    if len(ordered) < folds + 2:
        raise ValueError("not enough rows for requested walk-forward folds")

    step = max(1, len(ordered) // (folds + 1))
    fold_reports: list[RegimeFoldReport] = []
    metrics: list[ClassificationMetrics] = []
    directional = 0
    no_bet = 0

    for fold_index in range(1, folds + 1):
        test_start = step * fold_index
        test_end = step * (fold_index + 1) if fold_index < folds else len(ordered)
        train = ordered[:test_start]
        test = ordered[test_start:test_end]
        if not train or not test:
            continue
        report = evaluate_regime_persistence(train, test, window=window)
        metrics.append(
            ClassificationMetrics(
                accuracy=report.accuracy,
                total=report.directional_decisions,
                correct=report.correct,
            )
        )
        directional += report.directional_decisions
        no_bet += report.no_bet_decisions
        fold_reports.append(
            RegimeFoldReport(
                train_rows=len(train),
                test_rows=len(test),
                regime_persistence=metrics[-1],
                directional_decisions=report.directional_decisions,
                no_bet_decisions=report.no_bet_decisions,
            )
        )

    if not fold_reports:
        raise ValueError("no valid regime walk-forward folds were produced")
    total = sum(item.total for item in metrics)
    correct = sum(item.correct for item in metrics)
    aggregate = ClassificationMetrics(
        accuracy=correct / total if total else 0.0,
        total=total,
        correct=correct,
    )
    return RegimeWalkForwardReport(
        folds=tuple(fold_reports),
        regime_persistence=aggregate,
        directional_decisions=directional,
        no_bet_decisions=no_bet,
    )
