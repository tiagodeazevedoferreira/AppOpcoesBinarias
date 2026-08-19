from unittest.mock import Mock

import pytest

from app_opcoes_binarias.data.deriv_client import DerivPublicClient


def test_active_symbols_uses_current_request_shape() -> None:
    client = DerivPublicClient("wss://example.invalid")
    client.request = Mock(return_value={"active_symbols": []})  # type: ignore[method-assign]
    client.get_active_symbols()
    client.request.assert_called_once_with({"active_symbols": "brief", "req_id": 1})


def test_contracts_for_rejects_empty_symbol() -> None:
    client = DerivPublicClient("wss://example.invalid")
    with pytest.raises(ValueError, match="symbol is required"):
        client.get_contracts_for("")


def test_history_rejects_invalid_count() -> None:
    client = DerivPublicClient("wss://example.invalid")
    with pytest.raises(ValueError, match="greater than zero"):
        client.get_ticks_history("frxEURUSD", count=0)


def test_history_omits_unsupported_zero_subscribe_flag() -> None:
    client = DerivPublicClient("wss://example.invalid")
    client.request = Mock(return_value={"history": {"prices": [], "times": []}})  # type: ignore[method-assign]

    client.get_ticks_history("frxEURUSD", count=100, end=1766000000)

    client.request.assert_called_once_with(
        {
            "ticks_history": "frxEURUSD",
            "count": 100,
            "end": 1766000000,
            "style": "ticks",
            "req_id": 3,
        }
    )
