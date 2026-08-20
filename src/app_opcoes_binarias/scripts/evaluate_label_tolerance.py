from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.label_tolerance_evaluation import evaluate_tolerance_sweep


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate sensitivity to directional label tolerance.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument(
        "--tolerances",
        default="0,0.00001,0.00002,0.00005,0.0001,0.0002",
        help="Comma-separated price-unit tolerances.",
    )
    parser.add_argument("--output", default="artifacts/label_tolerance_report.json")
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")
    tolerances = tuple(float(value) for value in args.tolerances.split(",") if value.strip())
    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    reports = evaluate_tolerance_sweep(
        ticks,
        horizon_seconds=args.horizon,
        train_ratio=args.train_ratio,
        tolerances=tolerances,
    )
    payload = {
        "symbol": args.symbol,
        "horizon_seconds": args.horizon,
        "raw_tick_count": len(ticks),
        "reports": [asdict(report) for report in reports],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
