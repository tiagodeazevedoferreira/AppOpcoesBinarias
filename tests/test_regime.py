from app_opcoes_binarias.research.regime import label_regime


def test_regime_requires_complete_window() -> None:
    assert label_regime([1, 2], [100.0, 100.1], window=2) is None


def test_regime_detects_quiet_market() -> None:
    result = label_regime(list(range(6)), [100.0] * 6, window=3)
    assert result is not None
    assert result.regime == "QUIET"


def test_regime_detects_trend() -> None:
    result = label_regime(list(range(11)), [100.0 + i * 1.0 for i in range(11)], window=5)
    assert result is not None
    assert result.regime == "TREND"
