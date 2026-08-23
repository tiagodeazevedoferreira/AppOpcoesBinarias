from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.dataset import build_dataset, temporal_split
from app_opcoes_binarias.research.radical_edge import evaluate_radical, evaluate_radical_walk_forward


def _print_summary(report: dict) -> None:
    aggregate = report["walk_forward"]["aggregate"]
    print("\n=== RADICAL SELECTIVE EDGE ===")
    print(f"target_accuracy      : {aggregate['target_accuracy']:.4f}")
    print(f"walk-forward accuracy: {aggregate['accuracy']:.4f}")
    print(f"decisions            : {aggregate['total_decisions']}")
    print(f"decision rate        : {aggregate['decision_rate']:.6f}")
    print(f"coverage             : {aggregate['coverage_at_target']:.6f}")
    print(f"mean lower bound     : {aggregate['mean_lower_bound']:.4f}")
    for index, fold in enumerate(report["walk_forward"]["folds"], start=1):
        metrics = fold["metrics"]
        print(
            f"fold {index}: accuracy={metrics['accuracy']:.4f} "
            f"decisions={metrics['total_decisions']} "
            f"rate={metrics['decision_rate']:.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a selective state/motif model whose objective is >99% precision, not full coverage."
    )
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--min-evidence", type=int, default=100)
    parser.add_argument("--output", default="artifacts/radical_edge_report.json")
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")
    if not 0.50 < args.target_accuracy < 1.0:
        raise ValueError("target-accuracy must be between 0.50 and 1.0")
    if args.min_evidence < 2:
        raise ValueError("min-evidence must be at least 2")

    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    rows = build_dataset(ticks, horizon_seconds=args.horizon)
    train, test = temporal_split(rows, args.train_ratio)

    holdout = evaluate_radical(
        train,
        test,
        target_accuracy=args.target_accuracy,
        min_evidence=args.min_evidence,
    )
    walk_forward = evaluate_radical_walk_forward(
        rows,
        folds=args.folds,
        target_accuracy=args.target_accuracy,
        min_evidence=args.min_evidence,
    )

    non_overlapping = rows[:: max(1, args.horizon)]
    no_train, no_test = temporal_split(non_overlapping, args.train_ratio)
    non_overlapping_report = evaluate_radical(
        no_train,
        no_test,
        target_accuracy=args.target_accuracy,
        min_evidence=max(20, args.min_evidence // 2),
    )

    payload = {
        "symbol": args.symbol,
        "horizon_seconds": args.horizon,
        "train_ratio": args.train_ratio,
        "raw_tick_count": len(ticks),
        "dataset_rows": len(rows),
        "target_accuracy": args.target_accuracy,
        "min_evidence": args.min_evidence,
        "holdout": asdict(holdout),
        "walk_forward": asdict(walk_forward),
        "non_overlapping": asdict(non_overlapping_report),
        "interpretation": {
            "objective": "maximize precision of executed decisions while abstaining elsewhere",
            "promotion_rule": "only promote when every walk-forward fold has observed accuracy at or above target",
            "statistical_gate": "one-sided 99% Wilson lower bound on historical state purity",
            "coverage_is_secondary": True,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print_summary(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
