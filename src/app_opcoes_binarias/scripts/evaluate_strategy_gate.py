from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.dataset import build_dataset, temporal_split
from app_opcoes_binarias.research.evaluation import sample_non_overlapping
from app_opcoes_binarias.research.regime_evaluation import evaluate_regime_persistence
from app_opcoes_binarias.research.regime_walk_forward import evaluate_regime_walk_forward
from app_opcoes_binarias.research.selection import classify_strategy


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the research strategy gate without fitting predictive models.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--walk-forward-folds", type=int, default=5)
    parser.add_argument("--regime-window", type=int, default=60)
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")

    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    rows = build_dataset(ticks, horizon_seconds=args.horizon)
    train, test = temporal_split(rows, args.train_ratio)

    regime = evaluate_regime_persistence(train, test, window=args.regime_window)
    regime_walk = evaluate_regime_walk_forward(rows, folds=args.walk_forward_folds, window=args.regime_window)

    non_overlapping = sample_non_overlapping(rows, args.horizon)
    no_train, no_test = temporal_split(non_overlapping, args.train_ratio)
    non_overlap_regime = evaluate_regime_persistence(no_train, no_test, window=args.regime_window) if no_test else None

    total_oos_rows = sum(fold.test_rows for fold in regime_walk.folds)
    selection = classify_strategy(
        "regime_persistence",
        regime_walk.regime_persistence.accuracy,
        regime_walk.directional_decisions,
        regime_walk.directional_decisions / total_oos_rows if total_oos_rows else 0.0,
    )

    payload = {
        "symbol": args.symbol,
        "horizon_seconds": args.horizon,
        "raw_tick_count": len(ticks),
        "dataset_rows": len(rows),
        "regime_persistence": asdict(regime),
        "regime_walk_forward": asdict(regime_walk),
        "non_overlapping": {
            "rows": len(non_overlapping),
            "train_rows": len(no_train),
            "test_rows": len(no_test),
            "regime_persistence": asdict(non_overlap_regime) if non_overlap_regime else None,
        },
        "strategy_selection": asdict(selection),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
