from __future__ import annotations

import argparse
import json

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.bar_evaluation import evaluate_bar_sizes


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate temporal bar aggregation for short-horizon research.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--bar-sizes", default="1,5,15,30,60")
    args = parser.parse_args()
    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")
    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    sizes = tuple(int(value) for value in args.bar_sizes.split(",") if value.strip())
    reports = evaluate_bar_sizes(ticks, horizon_seconds=args.horizon, train_ratio=args.train_ratio, bar_sizes=sizes)
    print(json.dumps({"symbol": args.symbol, "horizon_seconds": args.horizon, "reports": [report.__dict__ for report in reports]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
