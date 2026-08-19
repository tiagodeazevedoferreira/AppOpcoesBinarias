from unittest.mock import Mock

import pytest

from app_opcoes_binarias.data.collector import collect_history_backfill
from app_opcoes_binarias.data.deriv_client import DerivPublicClient


def _history(symbol: str, epochs: list[int]) -> dict:
    return {
        "history": {
            "prices": [1.1 + i / 10000 for i in range(len(epochs))],
            "times": epochs,
        },
        "msg_type": "history",
        "symbol": symbol,
    }


def test_history_accepts_explicit_window() -> None:
    client = DerivPublicClient("wss://example.invalid")
    client.request = Mock(return_value=_history("frxEURUSD", [100, 101]))  # type: ignore[method-assign]

    result = client.get_ticks_history("frxEURUSD", count=2, start=99, end=101)

    assert result["history"]["times"] == [100, 101]
    client.request.assert_called_once_with(
        {
            "ticks_history": "frxEURUSD",
            "count": 2,
            "end": 101,
            "style": "ticks",
            "req_id": 3,
            "start": 99,
        }
    )


def test_backfill_walks_backwards_and_deduplicates() -> None:
    client = DerivPublicClient("wss://example.invalid")
    responses = [
        _history("frxEURUSD", [106, 107, 108]),
        _history("frxEURUSD", [103, 104, 105, 106]),
        _history("frxEURUSD", [100, 101, 102, 103]),
    ]
    client.get_ticks_history = Mock(side_effect=responses)  # type: ignore[method-assign]

    result = collect_history_backfill(
        client,
        "frxEURUSD",
        start=100,
        end=108,
        batch_size=1000,
        max_batches=5,
    )

    assert [tick["epoch"] for tick in result] == list(range(100, 109))
    assert client.get_ticks_history.call_args_list[1].kwargs["end"] == 105
    assert client.get_ticks_history.call_args_list[2].kwargs["end"] == 102


def test_backfill_rejects_invalid_range() -> None:
    client = DerivPublicClient("wss://example.invalid")
    with pytest.raises(ValueError, match="greater than or equal"):
        collect_history_backfill(client, "frxEURUSD", start=200, end=100)
