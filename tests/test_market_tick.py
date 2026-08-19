from decimal import Decimal

import pytest

from app_opcoes_binarias.data.models import MarketTick


def test_market_tick_normalizes_current_deriv_payload() -> None:
    tick = MarketTick.from_deriv(
        {
            "msg_type": "tick",
            "tick": {"symbol": "frxEURUSD", "epoch": 1724000000, "quote": 1.16642},
        }
    )
    assert tick.symbol == "frxEURUSD"
    assert tick.epoch == 1724000000
    assert tick.quote == Decimal("1.16642")
    assert tick.timestamp.tzinfo is not None


def test_market_tick_rejects_missing_tick() -> None:
    with pytest.raises(ValueError, match="missing 'tick'"):
        MarketTick.from_deriv({"msg_type": "error"})


def test_market_tick_accepts_underlying_symbol() -> None:
    tick = MarketTick.from_deriv(
        {"tick": {"underlying_symbol": "frxEURUSD", "epoch": 1724000000, "quote": "1.1"}}
    )
    assert tick.symbol == "frxEURUSD"
