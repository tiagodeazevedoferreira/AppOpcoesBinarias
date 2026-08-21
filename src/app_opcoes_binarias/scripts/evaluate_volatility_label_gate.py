# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.volatility_label_evaluation import evaluate_volatility_labels


MIN_BALANCED_ACCURACY = 0.55


def main() -> int:
    parser = argparse.ArgumentParser(description="Compact gate for volatility-normalized labels.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--volatility-window", type=int, default=60)
    parser.add_argument("--multipliers", default="0.5,1,1.5,2,2.5,3")
    parser.add_argument("--output", default="artifacts/volatility_label_gate.json")
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")

    multipliers = tuple(float(value) for value in args.multipliers.split(",") if value.strip())
    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    reports = evaluate_volatility_labels(
        ticks,
        horizon_seconds=args.horizon,
        train_ratio=args.train_ratio,
        multipliers=multipliers,
        volatility_window=args.volatility_window,
    )

    candidates = [
        report.multiplier
        for report in reports
        if report.non_overlapping_balanced_accuracy is not None
        and report.non_overlapping_balanced_accuracy >= MIN_BALANCED_ACCURACY
    ]
    rows = [
        {
            "multiplier": report.multiplier,
            "flat_ratio": round(report.flat_ratio, 6),
            "oos_balanced_accuracy": round(report.balanced_accuracy, 6),
            "non_overlapping_balanced_accuracy": round(
                report.non_overlapping_balanced_accuracy or 0.0, 6
            ),
            "non_overlapping_persistence_accuracy": round(
                report.non_overlapping_persistence_accuracy or 0.0, 6
            ),
            "status": "PASS" if report.multiplier in candidates else "REJECTED",
        }
        for report in reports
    ]
    payload = {
        "symbol": args.symbol,
        "horizon_seconds": args.horizon,
        "volatility_window": args.volatility_window,
        "minimum_non_overlapping_balanced_accuracy": MIN_BALANCED_ACCURACY,
        "candidates": candidates,
        "reports": [asdict(report) for report in reports],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("multiplier | flat_ratio | oos_balanced | non_overlap_balanced | non_overlap_persistence | status")
    for row in rows:
        print(
            f"{row['multiplier']:>10} | {row['flat_ratio']:.6f} | "
            f"{row['oos_balanced_accuracy']:.6f} | "
            f"{row['non_overlapping_balanced_accuracy']:.6f} | "
            f"{row['non_overlapping_persistence_accuracy']:.6f} | {row['status']}"
        )
    print(f"candidates={candidates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
