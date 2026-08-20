from __future__ import annotations

import argparse

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.signal_walk_forward import evaluate_signal_walk_forward


def main() -> int:
    parser = argparse.ArgumentParser(description="Compact walk-forward signal diagnostics.")
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
        ticks, horizon_seconds=args.horizon, folds=args.folds, lookbacks=lookbacks
    )
    print(f"horizon | {args.horizon}")
    print("lookback | folds | accuracies | mean | min | max | folds>=chance")
    for report in reports:
        accuracies = ",".join(f"{fold.accuracy:.4f}" for fold in report.folds)
        print(
            f"{report.lookback_seconds:8d} | {len(report.folds):5d} | {accuracies:29s} | "
            f"{report.mean_accuracy:.4f} | {report.min_accuracy:.4f} | "
            f"{report.max_accuracy:.4f} | {report.folds_at_or_above_chance}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
