from collections.abc import Mapping
from typing import Any

from .models import MarketTick


def normalize_tick(payload: Mapping[str, Any]) -> MarketTick:
    """Normalize a Deriv tick response into the application's canonical model."""
    tick = payload.get("tick")
    if not isinstance(tick, Mapping):
        raise TypeError("response does not contain a tick object")

    try:
        symbol = str(tick["symbol"])
        epoch = int(tick["epoch"])
        quote = float(tick["quote"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid Deriv tick payload") from exc

    if not symbol:
        raise ValueError("tick symbol cannot be empty")
    if epoch <= 0:
        raise ValueError("tick epoch must be positive")
    if quote <= 0:
        raise ValueError("tick quote must be positive")

    return MarketTick(symbol=symbol, epoch=epoch, quote=quote)
