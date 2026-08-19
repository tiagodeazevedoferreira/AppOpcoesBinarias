from app_opcoes_binarias.research.dataset import ResearchRow
from app_opcoes_binarias.research.horizon_evaluation import evaluate_horizons


def rows_for(horizon: int) -> list[ResearchRow]:
    rows: list[ResearchRow] = []
    for epoch in range(30):
        rows.append(
            ResearchRow(
                epoch,
                100.0 + epoch,
                0.01,
                0.02,
                0.001,
                "RISE" if epoch % 2 == 0 else "FALL",
                float(horizon),
                0.01,
                0.5,
            )
        )
    return rows


def test_evaluate_horizons_returns_sorted_reports() -> None:
    reports = evaluate_horizons({60: rows_for(60), 15: rows_for(15)})
    assert [report.horizon_seconds for report in reports] == [15, 60]
    assert all(report.test_rows > 0 for report in reports)
