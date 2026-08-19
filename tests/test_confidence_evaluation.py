from app_opcoes_binarias.research.confidence_evaluation import evaluate_softmax_confidence
from app_opcoes_binarias.research.dataset import ResearchRow


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


def test_confidence_evaluation_returns_complete_bins() -> None:
    train = [_row(i, "RISE" if i % 2 else "FALL") for i in range(20)]
    test = [_row(i + 20, "RISE" if i % 2 else "FALL") for i in range(10)]
    report = evaluate_softmax_confidence(train, test)

    assert report.usable_rows == 10
    assert sum(item.rows for item in report.bins) == 10
    assert report.mean_confidence > 0.0
    assert sum(report.prediction_distribution.values()) == 10
