from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class MarketTick:
    """Canonical market tick used by the application, independent of broker format."""

    symbol: str
    epoch: int
    quote: float

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.epoch, tz=timezone.utc)
