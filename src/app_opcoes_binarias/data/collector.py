from datetime import UTC, datetime
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
        "timestamp": datetime.fromtimestamp(epoch, tz=UTC).isoformat(),
        "quote": float(tick["quote"]),
    }


def _normalize_history(symbol: str, response: dict[str, Any]) -> list[dict[str, Any]]:
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
            "timestamp": datetime.fromtimestamp(int(epoch), tz=UTC).isoformat(),
            "quote": float(price),
        }
        for price, epoch in zip(prices, times, strict=True)
    ]


def collect_history(
    client: DerivPublicClient,
    symbol: str,
    count: int = 1000,
    *,
    start: int | None = None,
    end: int | str = "latest",
) -> list[dict[str, Any]]:
    """Fetch one historical tick window and normalize it."""
    response = client.get_ticks_history(symbol, count=count, start=start, end=end)
    return _normalize_history(symbol, response)


def collect_history_backfill(
    client: DerivPublicClient,
    symbol: str,
    *,
    start: int,
    end: int | str = "latest",
    batch_size: int = 1000,
    max_batches: int = 100,
) -> list[dict[str, Any]]:
    """Collect historical ticks backwards in bounded, deduplicated batches."""
    if start < 0:
        raise ValueError("start must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")
    if max_batches < 1:
        raise ValueError("max_batches must be greater than zero")
    if isinstance(end, int) and end < start:
        raise ValueError("end must be greater than or equal to start")

    collected: dict[int, dict[str, Any]] = {}
    cursor: int | str = end

    for _ in range(max_batches):
        batch = collect_history(client, symbol, count=batch_size, end=cursor)
        if not batch:
            break

        upper_bound = end if isinstance(end, int) else None
        for tick in batch:
            epoch = int(tick["epoch"])
            if epoch >= start and (upper_bound is None or epoch <= upper_bound):
                collected[epoch] = tick

        oldest = min(int(tick["epoch"]) for tick in batch)
        if oldest <= start:
            break
        next_cursor = oldest - 1
        if isinstance(cursor, int) and next_cursor >= cursor:
            raise RuntimeError("Historical cursor did not move backwards")
        cursor = next_cursor

    return [collected[epoch] for epoch in sorted(collected)]
