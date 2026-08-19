import pytest

from app_opcoes_binarias.research.selection import classify_strategy


def test_selection_requires_enough_decisions() -> None:
    result = classify_strategy("regime", 0.9, 10, 0.1)
    assert result.status == "RESEARCH_ONLY"


def test_selection_rejects_below_chance() -> None:
    result = classify_strategy("softmax", 0.49, 100, 0.5)
    assert result.status == "REJECTED"


def test_selection_promotes_only_strong_result() -> None:
    result = classify_strategy("candidate", 0.60, 100, 0.25)
    assert result.status == "PROMISING"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"accuracy": -0.1},
        {"accuracy": 1.1},
        {"total_decisions": -1},
        {"decision_rate": 1.1},
    ],
)
def test_selection_rejects_invalid_metrics(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        classify_strategy("x", 0.5, 100, 0.5, **kwargs)
