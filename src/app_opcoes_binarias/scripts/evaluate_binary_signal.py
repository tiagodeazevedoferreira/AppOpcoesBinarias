from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.binary_signal import (
    evaluate_binary_momentum,
    evaluate_binary_momentum_walk_forward,
)
from app_opcoes_binarias.research.dataset import build_dataset
from app_opcoes_binarias.research.evaluation import sample_non_overlapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate directional momentum as a binary signal.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--lookbacks", default="10,30,60")
    parser.add_argument("--quantiles", default="0,0.5,0.7,0.9")
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")

    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    rows = build_dataset(ticks, horizon_seconds=args.horizon)
    cut = int(len(rows) * args.train_ratio)
    train, test = rows[:cut], rows[cut:]
    lookbacks = [int(value) for value in args.lookbacks.split(",") if value]
    quantiles = [float(value) for value in args.quantiles.split(",") if value]

    holdout = [
        asdict(
            evaluate_binary_momentum(
                train,
                test,
                lookback_seconds=lookback,
                quantile=quantile,
            )
        )
        for lookback in lookbacks
        if lookback < args.horizon
        for quantile in quantiles
    ]

    walk_forward = []
    for lookback in lookbacks:
        if lookback >= args.horizon:
            continue
        for quantile in quantiles:
            folds = evaluate_binary_momentum_walk_forward(
                rows,
                lookback_seconds=lookback,
                quantile=quantile,
                folds=args.folds,
            )
            walk_forward.append(
                {
                    "lookback_seconds": lookback,
                    "quantile": quantile,
                    "folds": [asdict(report) for report in folds],
                    "mean_accuracy": sum(report.accuracy for report in folds) / len(folds),
                    "min_accuracy": min(report.accuracy for report in folds),
                    "max_accuracy": max(report.accuracy for report in folds),
                }
            )

    non_overlapping = sample_non_overlapping(rows, args.horizon)
    non_overlap_cut = int(len(non_overlapping) * args.train_ratio)
    non_overlap_train = non_overlapping[:non_overlap_cut]
    non_overlap_test = non_overlapping[non_overlap_cut:]
    non_overlap = [
        asdict(
            evaluate_binary_momentum(
                non_overlap_train,
                non_overlap_test,
                lookback_seconds=lookback,
                quantile=quantile,
            )
        )
        for lookback in lookbacks
        if lookback < args.horizon
        for quantile in quantiles
    ]

    payload = {
        "symbol": args.symbol,
        "horizon_seconds": args.horizon,
        "raw_tick_count": len(ticks),
        "dataset_rows": len(rows),
        "directional_test_rows": sum(row.label in {"RISE", "FALL"} for row in test),
        "holdout": holdout,
        "walk_forward": walk_forward,
        "non_overlapping": non_overlap,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
