from __future__ import annotations

import argparse

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.signal_threshold import evaluate_signal_thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate train-only thresholds for the 60s momentum signal.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--folds", type=int, default=20)
    parser.add_argument("--quantiles", default="0.50,0.60,0.70,0.80,0.90")
    args = parser.parse_args()
    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")
    quantiles = tuple(float(value) for value in args.quantiles.split(",") if value.strip())
    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    reports = evaluate_signal_thresholds(
        ticks,
        horizon_seconds=args.horizon,
        lookback_seconds=args.lookback,
        folds=args.folds,
        quantiles=quantiles,
    )
    print(f"horizon | {args.horizon} | lookback | {args.lookback} | folds | {args.folds}")
    print("quantile | mean | min | max | mean_decision_rate | directional_rows")
    for report in reports:
        print(
            f"{report.threshold_quantile:.2f} | {report.mean_accuracy:.4f} | "
            f"{report.min_accuracy:.4f} | {report.max_accuracy:.4f} | "
            f"{report.mean_decision_rate:.4f} | {report.total_directional_rows}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
