from unittest.mock import Mock

import pytest

from app_opcoes_binarias.data.tick_storage import TickStorage


def test_write_batch_uses_epoch_keys_and_deduplicates() -> None:
    store = Mock()
    storage = TickStorage(store)
    ticks = [
        {"symbol": "frxEURUSD", "epoch": 101, "quote": 1.2},
        {"symbol": "frxEURUSD", "epoch": 100, "quote": 1.1},
        {"symbol": "frxEURUSD", "epoch": 101, "quote": 1.21},
    ]

    persisted = storage.write_batch("frxEURUSD", ticks)

    assert persisted == 2
    store.update.assert_called_once_with(
        "market_ticks/frxEURUSD",
        {
            "100": ticks[1],
            "101": ticks[2],
        },
    )


def test_write_batch_empty_is_noop() -> None:
    store = Mock()
    storage = TickStorage(store)

    assert storage.write_batch("frxEURUSD", []) == 0
    store.update.assert_not_called()


def test_read_all_requires_symbol() -> None:
    storage = TickStorage(Mock())
    with pytest.raises(ValueError, match="symbol is required"):
        storage.read_all("")
