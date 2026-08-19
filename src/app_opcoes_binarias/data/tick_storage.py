from __future__ import annotations

from typing import Any, Iterable

from app_opcoes_binarias.data.firebase_store import FirebaseStore


class TickStorage:
    """Persist normalized ticks in a bounded Firebase collection."""

    def __init__(self, store: FirebaseStore, root: str = "market_ticks") -> None:
        self.store = store
        self.root = root.strip("/")

    def write_batch(self, symbol: str, ticks: Iterable[dict[str, Any]]) -> int:
        count = 0
        for tick in ticks:
            epoch = int(tick["epoch"])
            self.store.write(f"{self.root}/{symbol}/{epoch}", tick)
            count += 1
        return count
