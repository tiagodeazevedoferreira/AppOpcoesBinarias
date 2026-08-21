from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime, timedelta

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.collector import collect_history_backfill
from app_opcoes_binarias.data.deriv_client import DerivPublicClient
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.quality import assess_ticks
from app_opcoes_binarias.data.tick_storage import TickStorage

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expand persisted market history backwards from its oldest tick."
    )
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-batches", type=int, default=100)
    args = parser.parse_args()

    if args.hours <= 0:
        raise ValueError("hours must be greater than zero")
    if args.batch_size < 1:
        raise ValueError("batch-size must be greater than zero")
    if args.max_batches < 1:
        raise ValueError("max-batches must be greater than zero")
    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for persistence")

    logging.basicConfig(level=getattr(settings.log_level.upper(), "__missing__", logging.INFO))
    store = FirebaseStore(settings.firebase_database_url)
    storage = TickStorage(store)
    existing = storage.read_all(args.symbol)
    if not existing:
        raise RuntimeError("Cannot expand history backwards without existing persisted ticks")

    oldest_epoch = min(int(tick["epoch"]) for tick in existing)
    end_epoch = oldest_epoch - 1
    start_epoch = int(end_epoch - args.hours * 3600)

    logger.info(
        "Expanding %s backwards from %s to %s",
        args.symbol,
        datetime.fromtimestamp(end_epoch, tz=UTC).isoformat(),
        datetime.fromtimestamp(start_epoch, tz=UTC).isoformat(),
    )

    client = DerivPublicClient(settings.deriv_ws_url)
    client.connect()
    try:
        ticks = collect_history_backfill(
            client,
            args.symbol,
            start=start_epoch,
            end=end_epoch,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
    finally:
        client.close()

    quality = assess_ticks(ticks)
    logger.info("Market data quality: %s", quality)
    if not quality["valid_shape"] or not quality["ordered"]:
        raise RuntimeError("Collected market data failed structural quality checks")

    persisted = storage.write_batch(args.symbol, ticks)
    logger.info("Collected %s historical ticks and persisted %s", len(ticks), persisted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
