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


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate directional baselines on persisted market data.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--output", default="artifacts/baseline_report.json")
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")

    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    rows = build_dataset(ticks, horizon_seconds=args.horizon)
    train, test = temporal_split(rows, args.train_ratio)
    report = evaluate_baselines(train, test)

    non_overlapping = sample_non_overlapping(rows, args.horizon)
    non_overlap_train, non_overlap_test = temporal_split(non_overlapping, args.train_ratio)
    non_overlap_report = evaluate_baselines(non_overlap_train, non_overlap_test)

    payload = {
        "symbol": args.symbol,
        "horizon_seconds": args.horizon,
        "train_ratio": args.train_ratio,
        "raw_tick_count": len(ticks),
        "dataset_rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "labeled_rows": sum(row.label is not None for row in rows),
        "actual_horizon_seconds": {
            "min": min((row.actual_horizon_seconds for row in rows if row.actual_horizon_seconds is not None), default=None),
            "max": max((row.actual_horizon_seconds for row in rows if row.actual_horizon_seconds is not None), default=None),
        },
        "report": asdict(report),
        "non_overlapping": {
            "rows": len(non_overlapping),
            "train_rows": len(non_overlap_train),
            "test_rows": len(non_overlap_test),
            "report": asdict(non_overlap_report),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
