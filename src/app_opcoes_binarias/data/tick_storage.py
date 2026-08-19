from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app_opcoes_binarias.data.firebase_store import FirebaseStore


class TickStorage:
    """Persist and retrieve normalized ticks in Firebase."""

    def __init__(self, store: FirebaseStore, root: str = "market_ticks") -> None:
        self.store = store
        self.root = root.strip("/")

    def write_batch(self, symbol: str, ticks: Iterable[dict[str, Any]]) -> int:
        """Persist a batch atomically by epoch, making repeated runs idempotent."""
        records = list(ticks)
        if not records:
            return 0

        updates = {}
        for tick in records:
            epoch = int(tick["epoch"])
            updates[str(epoch)] = tick

        self.store.update(f"{self.root}/{symbol}", updates)
        return len(updates)

    def read_all(self, symbol: str) -> list[dict[str, Any]]:
        """Read all persisted ticks for a symbol in chronological order."""
        if not symbol:
            raise ValueError("symbol is required")
        value = self.store.read(f"{self.root}/{symbol}")
        if not value:
            return []
        if not isinstance(value, dict):
            raise TypeError("Firebase tick collection must be an object")
        ticks = [tick for tick in value.values() if isinstance(tick, dict)]
        return sorted(ticks, key=lambda tick: int(tick["epoch"]))
