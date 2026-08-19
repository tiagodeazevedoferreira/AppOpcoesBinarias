import pytest

from app_opcoes_binarias.research.features import ema, momentum, returns, rolling_volatility


def test_returns():
    result = returns([100.0, 101.0, 99.99])
    assert result[0] == pytest.approx(0.01)
    assert result[1] == pytest.approx(99.99 / 101.0 - 1.0)


def test_momentum():
    assert momentum([100.0, 101.0, 102.0], 2) == pytest.approx(0.02)


def test_ema_requires_enough_history():
    assert ema([1.0, 2.0], 3) is None
    assert ema([1.0, 2.0, 3.0], 3) == 2.0


def test_rolling_volatility_requires_window_history():
    assert rolling_volatility([1.0, 2.0], 2) == pytest.approx(0.0)
    assert rolling_volatility([1.0], 2) is None
