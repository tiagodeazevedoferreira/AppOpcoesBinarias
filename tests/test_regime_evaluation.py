from app_opcoes_binarias.research.dataset import ResearchRow
from app_opcoes_binarias.research.regime_evaluation import evaluate_regime_persistence


def make_rows() -> list[ResearchRow]:
    rows: list[ResearchRow] = []
    for epoch in range(20):
        rows.append(
            ResearchRow(
                epoch,
                100.0 + epoch,
                0.01,
                0.02,
                0.001,
                "RISE",
                60.0,
                0.01,
                0.5,
            )
        )
    return rows


def test_regime_strategy_needs_history_and_can_no_bet() -> None:
    report = evaluate_regime_persistence(make_rows()[:10], make_rows()[10:], window=60)
    assert report.directional_decisions == 0
    assert report.no_bet_decisions == 10
    assert report.decision_rate == 0.0


def test_regime_strategy_reports_directional_accuracy() -> None:
    rows = make_rows()
    report = evaluate_regime_persistence(rows[:10], rows[10:], window=5)
    assert report.trend_rows == report.directional_decisions
    assert report.correct == report.directional_decisions
    assert report.accuracy == 1.0
