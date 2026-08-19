from __future__ import annotations

from dataclasses import dataclass

from .dataset import ResearchRow
from .regime import label_regime


@dataclass(frozen=True)
class RegimeDecisionReport:
    window: int
    trend_rows: int
    directional_decisions: int
    no_bet_decisions: int
    correct: int
    accuracy: float
    decision_rate: float
    trend_direction_accuracy: float


def evaluate_regime_persistence(
    train: list[ResearchRow],
    test: list[ResearchRow],
    *,
    window: int = 60,
) -> RegimeDecisionReport:
    """Predict continuation only when the past window is classified as TREND.

    The regime is computed from observations available at the decision timestamp;
    no future test information is used to construct the signal.
    """
    ordered = sorted([*train, *test], key=lambda row: row.epoch)
    test_epochs = {row.epoch for row in test}
    decisions = 0
    no_bet = 0
    trend_rows = 0
    correct = 0
    trend_correct = 0

    for index, row in enumerate(ordered):
        if row.epoch not in test_epochs:
            continue
        history = ordered[: index + 1]
        if len(history) < window + 1:
            no_bet += 1
            continue
        regime = label_regime(
            [item.epoch for item in history],
            [item.quote for item in history],
            window=window,
        )
        if regime is None:
            no_bet += 1
            continue
        if regime.regime != "TREND" or regime.return_ratio == 0:
            no_bet += 1
            continue
        trend_rows += 1
        prediction = "RISE" if regime.return_ratio > 0 else "FALL"
        decisions += 1
        if row.label is not None and prediction == row.label:
            correct += 1
            trend_correct += 1

    usable_test = sum(1 for row in test if row.label is not None)
    return RegimeDecisionReport(
        window=window,
        trend_rows=trend_rows,
        directional_decisions=decisions,
        no_bet_decisions=no_bet,
        correct=correct,
        accuracy=correct / decisions if decisions else 0.0,
        decision_rate=decisions / usable_test if usable_test else 0.0,
        trend_direction_accuracy=trend_correct / trend_rows if trend_rows else 0.0,
    )
