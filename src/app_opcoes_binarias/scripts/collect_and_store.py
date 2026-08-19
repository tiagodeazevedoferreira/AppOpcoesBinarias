from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime, timedelta

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.collector import collect_history, collect_history_backfill
from app_opcoes_binarias.data.deriv_client import DerivPublicClient
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.quality import assess_ticks
from app_opcoes_binarias.data.tick_storage import TickStorage

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Deriv ticks and persist them in Firebase.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--hours", type=float, default=None, help="Backfill this many hours ending now.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-batches", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for persistence")
    if args.hours is not None and args.hours <= 0:
        raise ValueError("hours must be greater than zero")

    client = DerivPublicClient(settings.deriv_ws_url)
    client.connect()
    try:
        if args.hours is not None:
            end_epoch = int(datetime.now(tz=UTC).timestamp())
            start_epoch = int((datetime.now(tz=UTC) - timedelta(hours=args.hours)).timestamp())
            ticks = collect_history_backfill(
                client,
                args.symbol,
                start=start_epoch,
                end=end_epoch,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
            )
        else:
            ticks = collect_history(client, args.symbol, count=args.count)
    finally:
        client.close()

    quality = assess_ticks(ticks)
    logger.info("Market data quality: %s", quality)
    if not quality["valid_shape"] or not quality["ordered"]:
        raise RuntimeError("Collected market data failed structural quality checks")

    store = FirebaseStore(settings.firebase_database_url)
    persisted = TickStorage(store).write_batch(args.symbol, ticks)
    logger.info("Collected %s ticks and persisted %s", len(ticks), persisted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
