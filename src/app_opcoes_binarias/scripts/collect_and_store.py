from __future__ import annotations

import argparse
import logging

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.collector import collect_history
from app_opcoes_binarias.data.deriv_client import DerivPublicClient
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Deriv ticks and persist them in Firebase.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for persistence")

    client = DerivPublicClient(settings.deriv_ws_url)
    client.connect()
    try:
        ticks = collect_history(client, args.symbol, count=args.count)
    finally:
        client.close()

    store = FirebaseStore(settings.firebase_database_url)
    persisted = TickStorage(store).write_batch(args.symbol, ticks)
    logger.info("Collected %s ticks and persisted %s", len(ticks), persisted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
