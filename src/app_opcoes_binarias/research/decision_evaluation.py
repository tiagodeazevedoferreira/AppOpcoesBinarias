from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .baselines import ClassificationMetrics, accuracy
from .dataset import ResearchRow
from .decision import DecisionPolicy
from .model_evaluation import _features
from .models import SoftmaxClassifier


@dataclass(frozen=True)
class DecisionReport:
    total_rows: int
    directional_decisions: int
    no_bet_decisions: int
    decision_rate: float
    decision_accuracy: ClassificationMetrics
    no_bet_reasons: dict[str, int]


def evaluate_softmax_decisions(
    train: list[ResearchRow],
    test: list[ResearchRow],
    *,
    min_confidence: float = 0.55,
    min_margin: float = 0.10,
) -> DecisionReport:
    """Evaluate the decision layer on an out-of-sample Softmax model."""
    classifier = SoftmaxClassifier.fit(train)
    policy = DecisionPolicy(min_confidence=min_confidence, min_margin=min_margin)
    truth: list[str] = []
    predictions: list[str | None] = []
    reasons: Counter[str] = Counter()
    directional = 0
    skipped = 0

    for row in test:
        if row.label is None or _features(row) is None:
            skipped += 1
            continue
        probabilities = classifier.probabilities(row)
        if probabilities is None:
            skipped += 1
            continue
        decision = policy.decide(probabilities)
        reasons[decision.reason] += 1
        if decision.action == "BET":
            directional += 1
            truth.append(row.label)
            predictions.append(decision.direction)

    usable = len(test) - skipped
    no_bet = usable - directional
    return DecisionReport(
        total_rows=usable,
        directional_decisions=directional,
        no_bet_decisions=no_bet,
        decision_rate=directional / usable if usable else 0.0,
        decision_accuracy=accuracy(truth, predictions),
        no_bet_reasons=dict(reasons),
    )
