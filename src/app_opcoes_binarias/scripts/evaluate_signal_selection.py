from __future__ import annotations

import argparse

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.signal_selection import select_signal

LOOKBACK_MODE = "epoch_seconds_v2"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate statistical selection criteria for directional signals.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--lookbacks", default="1,2,5,10,30,60")
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")
    lookbacks = tuple(int(value) for value in args.lookbacks.split(",") if value.strip())
    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    selections = select_signal(
        ticks,
        horizon_seconds=args.horizon,
        folds=args.folds,
        lookbacks=lookbacks,
    )
    print(f"lookback_mode | {LOOKBACK_MODE}")
    print("lookback | mean | min | max | folds>=chance | ci95_low | ci95_high | pooled | balanced | target+ | status")
    for item in selections:
        print(
            f"{item.lookback_seconds:>8} | {item.mean_accuracy:.4f} | {item.min_accuracy:.4f} | "
            f"{item.max_accuracy:.4f} | {item.folds_at_or_above_chance:>12} | "
            f"{item.fold_ci95_low:.4f} | {item.fold_ci95_high:.4f} | {item.pooled_accuracy:.4f} | "
            f"{item.pooled_balanced_accuracy:.4f} | {item.positive_target_ratio:.4f} | {item.status}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
