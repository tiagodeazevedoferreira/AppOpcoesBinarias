import pytest

from app_opcoes_binarias.research.decision import DecisionPolicy


def test_directional_signal_can_produce_bet() -> None:
    decision = DecisionPolicy().decide({"RISE": 0.72, "FALL": 0.18, "FLAT": 0.10})
    assert decision.action == "BET"
    assert decision.direction == "RISE"
    assert decision.reason == "thresholds_satisfied"


def test_low_confidence_produces_no_bet() -> None:
    decision = DecisionPolicy().decide({"RISE": 0.50, "FALL": 0.30, "FLAT": 0.20})
    assert decision.action == "NO_BET"
    assert decision.reason == "confidence_below_threshold"


def test_small_margin_produces_no_bet() -> None:
    decision = DecisionPolicy().decide({"RISE": 0.56, "FALL": 0.52, "FLAT": 0.02})
    assert decision.action == "NO_BET"
    assert decision.reason == "margin_below_threshold"


def test_flat_top_class_produces_no_bet() -> None:
    decision = DecisionPolicy().decide({"FLAT": 0.70, "RISE": 0.20, "FALL": 0.10})
    assert decision.action == "NO_BET"
    assert decision.reason == "top_class_not_directional"


def test_empty_probabilities_produce_no_bet() -> None:
    decision = DecisionPolicy().decide({})
    assert decision.action == "NO_BET"
    assert decision.reason == "empty_probability_set"


def test_invalid_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        DecisionPolicy().decide({"RISE": 1.1, "FALL": 0.0})
