import pytest

from app_opcoes_binarias.research.features import (
    directional_consistency,
    ema,
    ema_distance,
    momentum,
    returns,
    rolling_volatility,
)


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


def test_ema_distance_uses_current_price_against_trailing_average():
    assert ema_distance([100.0, 102.0, 104.0], 3) == pytest.approx(104.0 / 102.0 - 1.0)
    assert ema_distance([100.0, 101.0], 3) is None


def test_directional_consistency_ignores_flat_changes():
    assert directional_consistency([1.0, 2.0, 2.0, 3.0], 4) == pytest.approx(1.0)
    assert directional_consistency([1.0, 2.0, 1.0, 2.0], 4) == pytest.approx(2.0 / 3.0)
    assert directional_consistency([1.0, 1.0, 1.0], 3) == pytest.approx(0.0)
    assert directional_consistency([1.0, 2.0], 3) is None
