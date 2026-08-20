from __future__ import annotations

import argparse
import json

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.signal_walk_forward import evaluate_signal_walk_forward


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate directional signal stability with chronological folds.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--lookbacks", default="1,2,5,10,30,60")
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")
    lookbacks = tuple(int(value) for value in args.lookbacks.split(",") if value.strip())
    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    reports = evaluate_signal_walk_forward(
        ticks,
        horizon_seconds=args.horizon,
        folds=args.folds,
        lookbacks=lookbacks,
    )
    payload = {
        "symbol": args.symbol,
        "horizon_seconds": args.horizon,
        "raw_tick_count": len(ticks),
        "reports": [
            {
                "lookback_seconds": report.lookback_seconds,
                "mean_accuracy": report.mean_accuracy,
                "min_accuracy": report.min_accuracy,
                "max_accuracy": report.max_accuracy,
                "folds_at_or_above_chance": report.folds_at_or_above_chance,
                "folds": [fold.__dict__ for fold in report.folds],
            }
            for report in reports
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
