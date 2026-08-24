from app_opcoes_binarias.research.dataset import ResearchRow
from app_opcoes_binarias.research.persistence_edge import diagnose_persistence_edge


def _row(index: int, quote: float, label: str = "RISE") -> ResearchRow:
    return ResearchRow(index, quote, None, None, None, label, 60, None, None)


def test_diagnostic_uses_observable_history_only():
    train = [_row(i, 100.0 + i, "RISE") for i in range(80)]
    test = [_row(i, 180.0 + i, "RISE") for i in range(20, 40)]
    report = diagnose_persistence_edge(
        train,
        test,
        horizon_seconds=60,
        lookback_seconds=60,
        min_evidence=2,
    )
    assert report.lookback_seconds == 60
    assert report.train_rows == 80
    assert all(cell.prediction in {"RISE", "FALL"} for cell in report.cells)


def test_diagnostic_can_find_repeated_observable_state():
    train = []
    for i in range(120):
        quote = 100.0 + i * 0.01
        train.append(_row(i, quote, "RISE"))
    test = [_row(i + 120, 101.2 + i * 0.01, "RISE") for i in range(20)]
    report = diagnose_persistence_edge(
        train,
        test,
        horizon_seconds=60,
        lookback_seconds=60,
        min_evidence=10,
    )
    assert report.cells
    assert report.top_cells[0].accuracy == 1.0
