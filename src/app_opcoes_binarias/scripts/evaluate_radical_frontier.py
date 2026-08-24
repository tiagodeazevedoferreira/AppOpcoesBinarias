from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.dataset import build_dataset, temporal_split
from app_opcoes_binarias.research.radical_frontier import evaluate_frontier


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the out-of-sample precision frontier before changing the radical 99% model."
    )
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="artifacts/radical_frontier_report.json")
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")

    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    rows = build_dataset(ticks, horizon_seconds=args.horizon)
    train, test = temporal_split(rows, args.train_ratio)
    holdout = evaluate_frontier(train, test)

    ordered = sorted(rows, key=lambda row: row.epoch)
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    chunk = len(ordered) // (args.folds + 1)
    if chunk == 0:
        raise ValueError("not enough rows for walk-forward evaluation")

    fold_reports = []
    for fold in range(1, args.folds + 1):
        train_end = chunk * fold
        test_end = chunk * (fold + 1) if fold < args.folds else len(ordered)
        fold_train = ordered[:train_end]
        fold_test = ordered[train_end:test_end]
        report = evaluate_frontier(fold_train, fold_test)
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": len(fold_train),
                "test_rows": len(fold_test),
                "best_accuracy": asdict(report.best_accuracy) if report.best_accuracy else None,
                "best_lower_bound": asdict(report.best_lower_bound) if report.best_lower_bound else None,
                "best_coverage_at_99": asdict(report.best_coverage_at_99) if report.best_coverage_at_99 else None,
            }
        )

    payload = {
        "symbol": args.symbol,
        "horizon_seconds": args.horizon,
        "train_ratio": args.train_ratio,
        "raw_tick_count": len(ticks),
        "dataset_rows": len(rows),
        "holdout": {
            "best_accuracy": asdict(holdout.best_accuracy) if holdout.best_accuracy else None,
            "best_lower_bound": asdict(holdout.best_lower_bound) if holdout.best_lower_bound else None,
            "best_coverage_at_99": asdict(holdout.best_coverage_at_99) if holdout.best_coverage_at_99 else None,
            "points": [asdict(point) for point in holdout.points],
        },
        "walk_forward": {"folds": fold_reports},
        "interpretation": {
            "purpose": "discover the conditional precision frontier before changing the promotion model",
            "anti_leakage": "training lower-bound thresholds are computed only from observations before each test fold",
            "promotion_unchanged": True,
            "views": ["state", "trend", "motif", "consensus"],
            "thresholds": [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.985, 0.99],
            "evidence_levels": [20, 50, 100],
            "min_views_levels": [1, 2],
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== RADICAL PRECISION FRONTIER ===")
    best = holdout.best_accuracy
    if best:
        print(
            f"holdout best accuracy={best.accuracy:.4f} decisions={best.decisions} "
            f"rate={best.decision_rate:.6f} view={best.view} "
            f"training_lower={best.training_lower_threshold:.3f} evidence={best.min_evidence} views={best.min_views}"
        )
    else:
        print("holdout: no eligible decisions")

    best_lower = holdout.best_lower_bound
    if best_lower:
        print(
            f"holdout best 99% lower-bound={best_lower.lower_bound:.4f} accuracy={best_lower.accuracy:.4f} "
            f"decisions={best_lower.decisions} rate={best_lower.decision_rate:.6f} view={best_lower.view}"
        )
    else:
        print("holdout: no eligible lower-bound candidate")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
