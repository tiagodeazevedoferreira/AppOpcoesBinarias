from app_opcoes_binarias.research.dataset import ResearchRow
from app_opcoes_binarias.research.models import NearestCentroidClassifier


def row(epoch: int, label: str | None, value: float) -> ResearchRow:
    return ResearchRow(epoch, value, value - 100.0, value - 100.0, 0.01, value - 100.0, 0.8, label, 60.0)


def test_model_fits_using_train_only_and_predicts_known_class() -> None:
    model = NearestCentroidClassifier.fit([
        row(1, "RISE", 101.0),
        row(2, "RISE", 102.0),
        row(3, "FALL", 99.0),
        row(4, "FALL", 98.0),
    ])

    assert model.predict(row(5, "RISE", 101.5)) == "RISE"
    assert model.predict(row(6, "FALL", 98.5)) == "FALL"


def test_model_ignores_incomplete_rows() -> None:
    model = NearestCentroidClassifier.fit([
        row(1, "RISE", 101.0),
        ResearchRow(2, "x", None, None, None, None, None, "FALL", 60.0),
    ])

    assert model.predict(row(3, None, 101.0)) == "RISE"
    assert model.predict(ResearchRow(4, 100.0, None, None, None, None, None, None, None)) is None
