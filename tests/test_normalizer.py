import pytest

from app_opcoes_binarias.data.normalizer import normalize_tick


def test_normalize_tick() -> None:
    tick = normalize_tick(
        {"tick": {"symbol": "frxEURUSD", "epoch": 1766000000, "quote": 1.16542}}
    )

    assert tick.symbol == "frxEURUSD"
    assert tick.epoch == 1766000000
    assert tick.quote == pytest.approx(1.16542)


def test_normalize_tick_rejects_missing_tick() -> None:
    with pytest.raises(ValueError, match="does not contain a tick"):
        normalize_tick({})


def test_normalize_tick_rejects_invalid_quote() -> None:
    with pytest.raises(ValueError, match="quote must be positive"):
        normalize_tick({"tick": {"symbol": "frxEURUSD", "epoch": 1, "quote": 0}})
