from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategySelection:
    name: str
    status: str
    accuracy: float
    minimum_accuracy: float
    total_decisions: int
    decision_rate: float
    rationale: str


def classify_strategy(
    name: str,
    observed_accuracy: float,
    observed_total_decisions: int,
    observed_decision_rate: float,
    *,
    minimum_accuracy: float = 0.55,
    minimum_decisions: int = 30,
    minimum_rate: float = 0.01,
    **overrides: Any,
) -> StrategySelection:
    """Classify an out-of-sample strategy conservatively for research progression."""
    if "accuracy" in overrides:
        observed_accuracy = overrides["accuracy"]
    if "total_decisions" in overrides:
        observed_total_decisions = overrides["total_decisions"]
    if "decision_rate" in overrides:
        observed_decision_rate = overrides["decision_rate"]
    unknown = set(overrides) - {"accuracy", "total_decisions", "decision_rate"}
    if unknown:
        raise TypeError(f"unexpected metric overrides: {sorted(unknown)}")

    accuracy = float(observed_accuracy)
    total_decisions = int(observed_total_decisions)
    decision_rate = float(observed_decision_rate)

    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be between 0 and 1")
    if total_decisions < 0:
        raise ValueError("total_decisions must be non-negative")
    if not 0.0 <= decision_rate <= 1.0:
        raise ValueError("decision_rate must be between 0 and 1")
    if not 0.0 <= minimum_accuracy <= 1.0:
        raise ValueError("minimum_accuracy must be between 0 and 1")
    if minimum_decisions < 1:
        raise ValueError("minimum_decisions must be positive")
    if not 0.0 <= minimum_rate <= 1.0:
        raise ValueError("minimum_rate must be between 0 and 1")

    if total_decisions < minimum_decisions:
        status = "RESEARCH_ONLY"
        rationale = "insufficient out-of-sample decisions"
    elif decision_rate < minimum_rate:
        status = "RESEARCH_ONLY"
        rationale = "decision rate is too low for practical evaluation"
    elif accuracy < 0.50:
        status = "REJECTED"
        rationale = "out-of-sample accuracy is below chance"
    elif accuracy < minimum_accuracy:
        status = "RESEARCH_ONLY"
        rationale = "accuracy is not strong enough for promotion"
    else:
        status = "PROMISING"
        rationale = "meets conservative out-of-sample promotion thresholds"

    return StrategySelection(
        name=name,
        status=status,
        accuracy=accuracy,
        minimum_accuracy=minimum_accuracy,
        total_decisions=total_decisions,
        decision_rate=decision_rate,
        rationale=rationale,
    )
