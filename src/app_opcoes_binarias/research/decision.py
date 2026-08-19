from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    action: str
    direction: str | None
    confidence: float
    margin: float
    reason: str


@dataclass(frozen=True)
class DecisionPolicy:
    min_confidence: float = 0.55
    min_margin: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0, 1]")
        if not 0.0 <= self.min_margin <= 1.0:
            raise ValueError("min_margin must be in [0, 1]")

    def decide(self, probabilities: Mapping[str, float]) -> Decision:
        if not probabilities:
            return Decision("NO_BET", None, 0.0, 0.0, "empty_probability_set")
        if any(value < 0.0 or value > 1.0 for value in probabilities.values()):
            raise ValueError("probabilities must be between 0 and 1")
        total = sum(probabilities.values())
        if total <= 0.0:
            raise ValueError("probabilities must have a positive sum")

        ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        (top_label, top_value), (_, second_value) = ordered[0], (
            ordered[1] if len(ordered) > 1 else (None, 0.0)
        )
        margin = top_value - second_value

        if top_label not in {"RISE", "FALL"}:
            return Decision("NO_BET", None, top_value, margin, "top_class_not_directional")
        if top_value < self.min_confidence:
            return Decision("NO_BET", None, top_value, margin, "confidence_below_threshold")
        if margin < self.min_margin:
            return Decision("NO_BET", None, top_value, margin, "margin_below_threshold")
        return Decision("BET", top_label, top_value, margin, "thresholds_satisfied")
