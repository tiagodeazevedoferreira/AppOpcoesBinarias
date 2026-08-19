from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.confidence_evaluation import evaluate_softmax_confidence
from app_opcoes_binarias.research.dataset import build_dataset, temporal_split
from app_opcoes_binarias.research.decision_evaluation import evaluate_softmax_decisions
from app_opcoes_binarias.research.evaluation import evaluate_baselines, sample_non_overlapping
from app_opcoes_binarias.research.horizon_evaluation import evaluate_horizons
from app_opcoes_binarias.research.model_evaluation import (
    evaluate_nearest_centroid,
    evaluate_softmax,
)
from app_opcoes_binarias.research.regime_evaluation import evaluate_regime_persistence
from app_opcoes_binarias.research.regime_walk_forward import evaluate_regime_walk_forward
from app_opcoes_binarias.research.stability import evaluate_walk_forward_stability
from app_opcoes_binarias.research.walk_forward import evaluate_walk_forward


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate leakage-safe research models out of sample.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--walk-forward-folds", type=int, default=5)
    parser.add_argument("--decision-min-confidence", type=float, default=0.55)
    parser.add_argument("--decision-min-margin", type=float, default=0.10)
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
    decisions = evaluate_softmax_decisions(
        train,
        test,
        min_confidence=args.decision_min_confidence,
        min_margin=args.decision_min_margin,
    )
    confidence = evaluate_softmax_confidence(train, test)
    regime_window = min(args.horizon, 60)
    regime_persistence = evaluate_regime_persistence(train, test, window=regime_window)
    regime_walk_forward = evaluate_regime_walk_forward(
        rows,
        folds=args.walk_forward_folds,
        window=regime_window,
    )

    non_overlapping = sample_non_overlapping(rows, args.horizon)
    non_overlap_train, non_overlap_test = temporal_split(non_overlapping, args.train_ratio)
    non_overlap_baseline = evaluate_baselines(non_overlap_train, non_overlap_test)
    non_overlap_nearest = evaluate_nearest_centroid(non_overlap_train, non_overlap_test) if non_overlap_test else None
    non_overlap_softmax = evaluate_softmax(non_overlap_train, non_overlap_test) if non_overlap_test else None
    non_overlap_decisions = (
        evaluate_softmax_decisions(
            non_overlap_train,
            non_overlap_test,
            min_confidence=args.decision_min_confidence,
            min_margin=args.decision_min_margin,
        )
        if non_overlap_test
        else None
    )
    non_overlap_confidence = (
        evaluate_softmax_confidence(non_overlap_train, non_overlap_test)
        if non_overlap_test
        else None
    )
    non_overlap_regime = (
        evaluate_regime_persistence(
            non_overlap_train,
            non_overlap_test,
            window=regime_window,
        )
        if non_overlap_test
        else None
    )
    walk_forward = evaluate_walk_forward(rows, folds=args.walk_forward_folds)
    stability = evaluate_walk_forward_stability(walk_forward)

    horizon_values = tuple(sorted({15, 30, 60, 120, 300, args.horizon}))
    rows_by_horizon = {horizon: build_dataset(ticks, horizon_seconds=horizon) for horizon in horizon_values}
    horizon_reports = evaluate_horizons(rows_by_horizon, train_ratio=args.train_ratio)

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
        "decision_policy": {
            "min_confidence": args.decision_min_confidence,
            "min_margin": args.decision_min_margin,
            "softmax": asdict(decisions),
        },
        "confidence": asdict(confidence),
        "horizon_comparison": [asdict(report) for report in horizon_reports],
        "regime_persistence": asdict(regime_persistence),
        "regime_walk_forward": asdict(regime_walk_forward),
        "non_overlapping": {
            "rows": len(non_overlapping),
            "train_rows": len(non_overlap_train),
            "test_rows": len(non_overlap_test),
            "baseline": asdict(non_overlap_baseline),
            "nearest_centroid": asdict(non_overlap_nearest) if non_overlap_nearest else None,
            "softmax": asdict(non_overlap_softmax) if non_overlap_softmax else None,
            "decision_policy": asdict(non_overlap_decisions) if non_overlap_decisions else None,
            "confidence": asdict(non_overlap_confidence) if non_overlap_confidence else None,
            "regime_persistence": asdict(non_overlap_regime) if non_overlap_regime else None,
        },
        "walk_forward": asdict(walk_forward),
        "walk_forward_stability": asdict(stability),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
