from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any


REQUIRED_TICK_FIELDS = frozenset({"symbol", "epoch", "quote"})


def assess_ticks(ticks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return deterministic quality metadata for a normalized tick batch."""
    records = list(ticks)
    epochs = [int(tick["epoch"]) for tick in records]

    ordered = epochs == sorted(epochs)
    unique = len(epochs) == len(set(epochs))
    valid_shape = all(REQUIRED_TICK_FIELDS.issubset(tick.keys()) for tick in records)

    first_epoch = epochs[0] if epochs else None
    last_epoch = epochs[-1] if epochs else None

    return {
        "record_count": len(records),
        "ordered": ordered,
        "unique_epochs": unique,
        "valid_shape": valid_shape,
        "first_epoch": first_epoch,
        "last_epoch": last_epoch,
        "first_timestamp": datetime.fromtimestamp(first_epoch, tz=UTC).isoformat() if first_epoch is not None else None,
        "last_timestamp": datetime.fromtimestamp(last_epoch, tz=UTC).isoformat() if last_epoch is not None else None,
    }
