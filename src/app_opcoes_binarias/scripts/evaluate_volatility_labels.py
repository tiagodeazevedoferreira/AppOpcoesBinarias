from __future__ import annotations

import argparse
import json
from pathlib import Path

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.volatility_label_evaluation import evaluate_volatility_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate volatility-normalized directional labels.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--volatility-window", type=int, default=60)
    parser.add_argument("--multipliers", default="0.5,1,1.5,2,2.5,3")
    parser.add_argument("--output", default="artifacts/volatility_label_report.json")
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")
    multipliers = tuple(float(value) for value in args.multipliers.split(",") if value.strip())
    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    reports = evaluate_volatility_labels(
        ticks,
        horizon_seconds=args.horizon,
        train_ratio=args.train_ratio,
        multipliers=multipliers,
        volatility_window=args.volatility_window,
    )
    payload = {
        "symbol": args.symbol,
        "horizon_seconds": args.horizon,
        "volatility_window": args.volatility_window,
        "raw_tick_count": len(ticks),
        "reports": [report.__dict__ for report in reports],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
