from app_opcoes_binarias.research.signal_selection import select_signal


def ticks_for_signal() -> list[dict[str, float | int]]:
    return [{"epoch": index, "quote": 100.0 + index * 0.01} for index in range(600)]


def test_signal_selection_returns_all_requested_lookbacks() -> None:
    reports = select_signal(
        ticks_for_signal(),
        horizon_seconds=10,
        folds=5,
        lookbacks=(1, 2, 5),
    )
    assert [report.lookback_seconds for report in reports] == [1, 2, 5]
    assert all(0.0 <= report.pooled_balanced_accuracy <= 1.0 for report in reports)
    assert all(0.0 <= report.positive_target_ratio <= 1.0 for report in reports)


def test_signal_selection_rejects_invalid_configuration() -> None:
    try:
        select_signal(ticks_for_signal(), horizon_seconds=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid horizon to raise ValueError")
