import pytest

from app_opcoes_binarias.scripts.expand_historical_data import _expansion_window


def test_expansion_window_ends_before_oldest_tick() -> None:
    start, end = _expansion_window([{"epoch": 1_000}, {"epoch": 2_000}], 24)

    assert end == 999
    assert start == 999 - 24 * 3600


def test_expansion_window_requires_existing_history() -> None:
    with pytest.raises(RuntimeError, match="existing persisted ticks"):
        _expansion_window([], 24)


def test_expansion_window_requires_positive_hours() -> None:
    with pytest.raises(ValueError, match="hours must be greater than zero"):
        _expansion_window([{"epoch": 1_000}], 0)
