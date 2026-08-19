from datetime import datetime, timezone
from typing import Any

from app_opcoes_binarias.data.deriv_client import DerivPublicClient


def normalize_tick(response: dict[str, Any]) -> dict[str, Any]:
    """Convert a Deriv tick response into our canonical tick representation."""
    tick = response.get("tick")
    if not tick:
        raise ValueError("Response does not contain a tick")

    epoch = int(tick["epoch"])
    return {
        "symbol": tick["symbol"],
        "epoch": epoch,
        "timestamp": datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(),
        "quote": float(tick["quote"]),
    }


def collect_history(client: DerivPublicClient, symbol: str, count: int = 1000) -> list[dict[str, Any]]:
    """Fetch historical ticks and normalize them into stable application records."""
    response = client.get_ticks_history(symbol, count=count)
    history = response.get("history")
    if not history:
        return []

    prices = history.get("prices", [])
    times = history.get("times", [])
    if len(prices) != len(times):
        raise ValueError("Deriv returned mismatched price/time arrays")

    return [
        {
            "symbol": symbol,
            "epoch": int(epoch),
            "timestamp": datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat(),
            "quote": float(price),
        }
        for price, epoch in zip(prices, times, strict=True)
    ]
