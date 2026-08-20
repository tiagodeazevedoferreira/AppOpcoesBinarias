from __future__ import annotations

import argparse
import bisect
from collections import Counter

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit label-persistence leakage against an observable momentum baseline.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    args = parser.parse_args()

    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 < args.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")

    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    ordered = sorted((int(row["epoch"]), float(row["quote"])) for row in ticks)
    epochs = [epoch for epoch, _ in ordered]
    prices = [price for _, price in ordered]
    split_index = int(len(ordered) * args.train_ratio)
    split_epoch = epochs[split_index]

    rows: list[tuple[int, str, float | None]] = []
    for i, epoch in enumerate(epochs):
        future_index = bisect.bisect_left(epochs, epoch + args.horizon, i + 1)
        if future_index >= len(epochs) or prices[i] == 0 or prices[future_index] == 0:
            continue
        target = "RISE" if prices[future_index] > prices[i] else "FALL" if prices[future_index] < prices[i] else "FLAT"
        previous_index = bisect.bisect_right(epochs, epoch - args.horizon, 0, i + 1) - 1
        momentum = None
        if previous_index >= 0 and prices[previous_index] != 0:
            previous_return = prices[i] / prices[previous_index] - 1.0
            momentum = previous_return
        rows.append((epoch, target, momentum))

    train = [row for row in rows if row[0] < split_epoch]
    holdout = [row for row in rows if row[0] >= split_epoch]

    label_persistence_correct = 0
    previous_label = train[-1][1] if train else None
    for _, target, _ in holdout:
        if previous_label is not None and previous_label == target:
            label_persistence_correct += 1
        previous_label = target

    momentum_pairs = [row for row in holdout if row[2] is not None and row[2] != 0]
    momentum_correct = sum((momentum > 0) == (target == "RISE") for _, target, momentum in momentum_pairs)

    non_overlap: list[tuple[int, str, float]] = []
    next_allowed: int | None = None
    for row in holdout:
        epoch, target, momentum = row
        if momentum is None or momentum == 0:
            continue
        if next_allowed is None or epoch >= next_allowed:
            non_overlap.append(row)  # type: ignore[arg-type]
            next_allowed = epoch + args.horizon

    non_overlap_correct = sum((momentum > 0) == (target == "RISE") for _, target, momentum in non_overlap)
    target_counts = Counter(target for _, target, _ in holdout)

    print(f"symbol | {args.symbol}")
    print(f"horizon | {args.horizon} | train_ratio | {args.train_ratio:.2f}")
    print(f"ticks | {len(ticks)} | rows | {len(rows)} | train_rows | {len(train)} | holdout_rows | {len(holdout)}")
    print(f"holdout_target_distribution | {dict(target_counts)}")
    print(
        "label_persistence_invalidation | accuracy | "
        f"{label_persistence_correct / len(holdout) if holdout else 0.0:.4f} | "
        "interpretation | uses prior future label in dense timestamps"
    )
    print(
        "observable_momentum_baseline | accuracy | "
        f"{momentum_correct / len(momentum_pairs) if momentum_pairs else 0.0:.4f} | "
        f"rows | {len(momentum_pairs)}"
    )
    print(
        "observable_momentum_non_overlapping | accuracy | "
        f"{non_overlap_correct / len(non_overlap) if non_overlap else 0.0:.4f} | "
        f"rows | {len(non_overlap)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
