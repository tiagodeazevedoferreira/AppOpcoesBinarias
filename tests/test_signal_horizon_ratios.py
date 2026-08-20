from app_opcoes_binarias.scripts.evaluate_signal_horizon_ratios import _evaluate


def test_evaluate_signal_horizon_ratio_uses_real_epochs() -> None:
    ticks = [{"epoch": i, "quote": 100.0 + i} for i in range(400)]
    mean, minimum, maximum, at_chance, folds = _evaluate(
        ticks,
        horizon=20,
        lookback=10,
        folds=4,
    )
    assert len(folds) == 4
    assert 0.0 <= minimum <= mean <= maximum <= 1.0
    assert 0 <= at_chance <= 4
