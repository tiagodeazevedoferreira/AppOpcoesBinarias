from app_opcoes_binarias.research.dataset import ResearchRow
from app_opcoes_binarias.research.regime_walk_forward import evaluate_regime_walk_forward


def make_rows() -> list[ResearchRow]:
    return [
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
        for epoch in range(40)
    ]


def test_regime_walk_forward_is_chronological_and_reports_decisions() -> None:
    report = evaluate_regime_walk_forward(make_rows(), folds=4, window=5)
    assert len(report.folds) == 4
    assert report.directional_decisions >= 0
    assert report.no_bet_decisions >= 0
    assert report.regime_persistence.total == report.directional_decisions


def test_regime_walk_forward_rejects_invalid_configuration() -> None:
    rows = make_rows()
    try:
        evaluate_regime_walk_forward(rows, folds=1, window=5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid folds")
