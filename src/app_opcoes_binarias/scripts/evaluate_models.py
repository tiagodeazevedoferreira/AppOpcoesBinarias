from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.dataset import build_dataset, temporal_split
from app_opcoes_binarias.research.evaluation import evaluate_baselines, sample_non_overlapping
from app_opcoes_binarias.research.model_evaluation import (
    evaluate_nearest_centroid,
    evaluate_softmax,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate leakage-safe research models out of sample.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--output", default="artifacts/model_report.json")
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")

    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    rows = build_dataset(ticks, horizon_seconds=args.horizon)
    train, test = temporal_split(rows, args.train_ratio)
    baseline = evaluate_baselines(train, test)
    nearest_centroid = evaluate_nearest_centroid(train, test)
    softmax = evaluate_softmax(train, test)

    non_overlapping = sample_non_overlapping(rows, args.horizon)
    non_overlap_train, non_overlap_test = temporal_split(non_overlapping, args.train_ratio)
    non_overlap_baseline = evaluate_baselines(non_overlap_train, non_overlap_test)
    non_overlap_nearest = evaluate_nearest_centroid(non_overlap_train, non_overlap_test) if non_overlap_test else None
    non_overlap_softmax = evaluate_softmax(non_overlap_train, non_overlap_test) if non_overlap_test else None

    payload = {
        "symbol": args.symbol,
        "horizon_seconds": args.horizon,
        "train_ratio": args.train_ratio,
        "raw_tick_count": len(ticks),
        "dataset_rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "baseline": asdict(baseline),
        "nearest_centroid": asdict(nearest_centroid),
        "softmax": asdict(softmax),
        "non_overlapping": {
            "rows": len(non_overlapping),
            "train_rows": len(non_overlap_train),
            "test_rows": len(non_overlap_test),
            "baseline": asdict(non_overlap_baseline),
            "nearest_centroid": asdict(non_overlap_nearest) if non_overlap_nearest else None,
            "softmax": asdict(non_overlap_softmax) if non_overlap_softmax else None,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
