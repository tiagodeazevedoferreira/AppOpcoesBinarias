from app_opcoes_binarias.research.dataset import ResearchRow
from app_opcoes_binarias.research.observable_baselines import evaluate_momentum_baseline


def row(epoch: int, quote: float, label: str | None) -> ResearchRow:
    return ResearchRow(epoch, quote, None, None, None, label, None)


def test_momentum_baseline_uses_only_quotes_available_at_observation() -> None:
    train = [row(0, 100.0, "RISE"), row(1, 101.0, "RISE")]
    test = [row(60, 102.0, "RISE"), row(120, 101.0, "FALL")]

    report = evaluate_momentum_baseline(train, test, lookback_seconds=60)

    assert report.total == 2
    assert report.correct == 2


def test_momentum_baseline_requires_positive_lookback() -> None:
    try:
        evaluate_momentum_baseline([row(0, 100.0, "RISE")], [row(60, 101.0, "RISE")], 0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
