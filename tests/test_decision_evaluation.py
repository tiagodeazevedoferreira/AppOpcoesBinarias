from app_opcoes_binarias.research.dataset import ResearchRow
from app_opcoes_binarias.research.decision_evaluation import evaluate_softmax_decisions


def _row(epoch: int, label: str) -> ResearchRow:
    return ResearchRow(
        epoch=epoch,
        quote=100.0 + epoch / 10.0,
        return_1=0.01 if epoch % 2 else -0.01,
        momentum_2=0.02 if epoch % 2 else -0.02,
        volatility_5=0.01,
        label=label,
        actual_horizon_seconds=60,
        ema_distance_10=0.01 if epoch % 2 else -0.01,
        directional_consistency_5=1.0 if epoch % 2 else -1.0,
    )


def test_softmax_decision_evaluation_reports_explicit_no_bet() -> None:
    train = [_row(i, "RISE" if i % 2 else "FALL") for i in range(30)]
    test = [_row(i, "RISE" if i % 2 else "FALL") for i in range(30, 40)]
    report = evaluate_softmax_decisions(train, test)

    assert report.total_rows == len(test)
    assert report.directional_decisions + report.no_bet_decisions == report.total_rows
    assert 0.0 <= report.decision_rate <= 1.0
    assert report.no_bet_reasons
