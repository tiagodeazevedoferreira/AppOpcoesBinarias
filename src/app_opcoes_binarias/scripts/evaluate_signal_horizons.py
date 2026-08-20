from __future__ import annotations

import argparse

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.signal_walk_forward import evaluate_signal_walk_forward


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare directional signal stability across horizons.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--horizons", default="30,60,120,300")
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)

    print("horizon | lookback | folds | mean | min | max | folds>=chance")
    for horizon in horizons:
        if args.lookback >= horizon:
            print(f"{horizon:>7} | {args.lookback:>8} | INVALID: lookback must be < horizon")
            continue
        report = evaluate_signal_walk_forward(
            ticks,
            horizon_seconds=horizon,
            folds=args.folds,
            lookbacks=(args.lookback,),
        )[0]
        accuracies = ",".join(f"{fold.accuracy:.4f}" for fold in report.folds)
        print(
            f"{horizon:>7} | {args.lookback:>8} | {args.folds:>5} | "
            f"{report.mean_accuracy:.4f} | {report.min_accuracy:.4f} | "
            f"{report.max_accuracy:.4f} | {report.folds_at_or_above_chance:>12}"
        )
        print(f"  folds: {accuracies}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
