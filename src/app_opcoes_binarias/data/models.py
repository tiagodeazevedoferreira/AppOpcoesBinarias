from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class MarketTick(BaseModel):
    """Canonical market tick used throughout the application."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=2, max_length=30)
    epoch: int = Field(ge=0)
    quote: Decimal

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.epoch, tz=UTC)

    @classmethod
    def from_deriv(cls, payload: dict) -> "MarketTick":
        tick = payload.get("tick")
        if not isinstance(tick, dict):
            raise TypeError("Deriv tick payload is missing 'tick'")
        symbol = tick.get("symbol") or tick.get("underlying_symbol")
        if not isinstance(symbol, str):
            raise TypeError("Deriv tick payload is missing a valid symbol")
        epoch = tick.get("epoch")
        quote = tick.get("quote")
        if not isinstance(epoch, (int, float)) or not isinstance(quote, (int, float, str)):
            raise TypeError("Deriv tick payload contains invalid epoch or quote")
        return cls(symbol=symbol, epoch=int(epoch), quote=Decimal(str(quote)))
