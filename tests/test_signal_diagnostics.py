from app_opcoes_binarias.research.signal_diagnostics import evaluate_signal_diagnostics


def test_signal_diagnostics_return_requested_lookbacks() -> None:
    ticks = [{"epoch": i, "quote": 100.0 + i * 0.01} for i in range(200)]
    reports = evaluate_signal_diagnostics(ticks, horizon_seconds=60, lookbacks=(1, 5))
    assert [report.lookback_seconds for report in reports] == [1, 5]
    assert all(report.test_rows > 0 for report in reports)
