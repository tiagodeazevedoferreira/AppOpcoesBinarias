from app_opcoes_binarias.research.binary_signal import (
    evaluate_binary_momentum,
    evaluate_binary_momentum_walk_forward,
)
from app_opcoes_binarias.research.dataset import ResearchRow


def row(epoch: int, quote: float, label: str | None) -> ResearchRow:
    return ResearchRow(epoch, quote, None, None, None, label, None)


def test_binary_momentum_ignores_flat_outcomes() -> None:
    train = [row(0, 100.0, "RISE"), row(60, 101.0, "RISE"), row(120, 102.0, "FALL")]
    test = [row(180, 103.0, "RISE"), row(240, 104.0, "FLAT"), row(300, 105.0, "RISE")]

    report = evaluate_binary_momentum(train, test, lookback_seconds=60)

    assert report.decisions == 2
    assert report.correct == 2
    assert report.accuracy == 1.0


def test_binary_threshold_is_fit_from_train_only() -> None:
    train = [row(0, 100.0, "RISE"), row(60, 101.0, "RISE"), row(120, 103.0, "RISE")]
    test = [row(180, 104.0, "RISE"), row(240, 110.0, "RISE")]

    report = evaluate_binary_momentum(train, test, lookback_seconds=60, quantile=1.0)

    assert 0.019 < report.threshold < 0.021
    assert report.decisions == 1
    assert report.correct == 1


def test_binary_threshold_does_not_use_test_outlier() -> None:
    train = [row(0, 100.0, "RISE"), row(60, 101.0, "RISE"), row(120, 103.0, "RISE")]
    test = [row(180, 104.0, "RISE"), row(240, 110.0, "RISE")]

    report = evaluate_binary_momentum(train, test, lookback_seconds=60, quantile=1.0)

    assert 0.019 < report.threshold < 0.021


def test_binary_walk_forward_produces_requested_folds() -> None:
    rows = [row(epoch, 100.0 + epoch / 60.0, "RISE") for epoch in range(0, 600, 60)]

    reports = evaluate_binary_momentum_walk_forward(rows, lookback_seconds=60, quantile=0.7, folds=3)

    assert len(reports) == 3
    assert all(report.decisions >= 0 for report in reports)
