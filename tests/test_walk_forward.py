import pytest

from app_opcoes_binarias.research.dataset import ResearchRow
from app_opcoes_binarias.research.walk_forward import evaluate_walk_forward


def _row(epoch: int, label: str) -> ResearchRow:
    return ResearchRow(
        epoch,
        100.0 + epoch / 100.0,
        0.001,
        0.002,
        0.01,
        label,
        60.0,
        0.001,
        0.5,
    )


def test_walk_forward_produces_requested_folds() -> None:
    rows = [_row(i, "RISE" if i % 2 else "FALL") for i in range(40)]
    report = evaluate_walk_forward(rows, folds=4)

    assert len(report.folds) == 4
    assert report.nearest_centroid.total > 0
    assert report.softmax.total > 0


def test_walk_forward_rejects_too_few_folds() -> None:
    rows = [_row(i, "RISE") for i in range(10)]
    with pytest.raises(ValueError, match="folds"):
        evaluate_walk_forward(rows, folds=1)
