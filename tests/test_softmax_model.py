from app_opcoes_binarias.research.dataset import ResearchRow
from app_opcoes_binarias.research.models import SoftmaxClassifier


def row(epoch: int, label: str, value: float) -> ResearchRow:
    return ResearchRow(
        epoch,
        value,
        value - 100.0,
        value - 100.0,
        0.01,
        label,
        60.0,
        value - 100.0,
        0.8,
    )


def test_softmax_predicts_known_class_and_probabilities_sum_to_one() -> None:
    model = SoftmaxClassifier.fit(
        [row(1, "RISE", 101.0), row(2, "RISE", 102.0), row(3, "FALL", 99.0), row(4, "FALL", 98.0)],
        epochs=100,
    )
    probabilities = model.probabilities(row(5, "RISE", 101.5))

    assert probabilities is not None
    assert abs(sum(probabilities.values()) - 1.0) < 1e-9
    assert model.predict(row(5, "RISE", 101.5)) == "RISE"


def test_softmax_returns_none_for_incomplete_features() -> None:
    model = SoftmaxClassifier.fit([row(1, "RISE", 101.0), row(2, "FALL", 99.0)], epochs=10)
    incomplete = ResearchRow(3, 100.0, None, None, None, None, None)

    assert model.predict(incomplete) is None
    assert model.probabilities(incomplete) is None
