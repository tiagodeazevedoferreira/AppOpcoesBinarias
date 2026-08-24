from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage
from app_opcoes_binarias.research.dataset import build_dataset, temporal_split
from app_opcoes_binarias.research.persistence_edge import diagnose_persistence_edge


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find observable state pockets where persistence is unusually reliable."
    )
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizons", default="60,120,300")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--min-evidence", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--output", default="artifacts/persistence_edge_diagnostic.json")
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")
    horizons = tuple(sorted({int(value.strip()) for value in args.horizons.split(",") if value.strip()}))
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must contain positive integers")
    if not 0 < args.train_ratio < 1:
        raise ValueError("train-ratio must be between 0 and 1")
    if args.min_evidence < 2:
        raise ValueError("min-evidence must be at least 2")

    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    reports: list[dict] = []
    for horizon in horizons:
        rows = build_dataset(ticks, horizon_seconds=horizon)
        train, test = temporal_split(rows, args.train_ratio)
        lookbacks = tuple(sorted({max(15, horizon // 4), max(30, horizon // 2), horizon}))
        for lookback in lookbacks:
            diagnostic = diagnose_persistence_edge(
                train,
                test,
                horizon_seconds=horizon,
                lookback_seconds=lookback,
                min_evidence=args.min_evidence,
                top_n=args.top_n,
            )
            reports.append(asdict(diagnostic))

    reports.sort(
        key=lambda report: (
            report["top_cells"][0]["lower_bound"] if report["top_cells"] else 0.0,
            report["top_cells"][0]["accuracy"] if report["top_cells"] else 0.0,
        ),
        reverse=True,
    )

    payload = {
        "symbol": args.symbol,
        "raw_tick_count": len(ticks),
        "train_ratio": args.train_ratio,
        "min_evidence": args.min_evidence,
        "objective": "discover observable state pockets before applying a >99% selective gate",
        "important_baseline_warning": "label persistence is not executable because it uses a previous future label; use only observable price history",
        "reports": reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("=== OBSERVABLE PERSISTENCE EDGE DIAGNOSTIC ===")
    for report in reports:
        best = report["top_cells"][0] if report["top_cells"] else None
        if best:
            print(
                f"horizon={report['horizon_seconds']} lookback={report['lookback_seconds']} "
                f"best_accuracy={best['accuracy']:.4f} lower={best['lower_bound']:.4f} "
                f"evidence={best['evidence']} prediction={best['prediction']}"
            )
        else:
            print(
                f"horizon={report['horizon_seconds']} lookback={report['lookback_seconds']} "
                "no cells met evidence threshold"
            )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
