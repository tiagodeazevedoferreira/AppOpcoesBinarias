from __future__ import annotations

import argparse
import bisect
import math

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage


def _wilson(correct: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen momentum-threshold holdout evaluation.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--quantile", type=float, default=0.70)
    args = parser.parse_args()
    if args.horizon <= 0 or args.lookback <= 0 or args.lookback > args.horizon:
        raise ValueError("lookback and horizon must be positive with lookback <= horizon")
    if not 0 < args.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < args.quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")

    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)
    ordered = sorted((int(row["epoch"]), float(row["quote"])) for row in ticks)
    split_epoch = ordered[int(len(ordered) * args.train_ratio)][0]
    epochs = [epoch for epoch, _ in ordered]
    prices = [price for _, price in ordered]

    pairs: list[tuple[int, float, float]] = []
    for i, epoch in enumerate(epochs):
        lookback_index = bisect.bisect_right(epochs, epoch - args.lookback, 0, i + 1) - 1
        future_index = bisect.bisect_left(epochs, epoch + args.horizon, i + 1)
        if lookback_index < 0 or future_index >= len(epochs):
            continue
        current = prices[i]
        previous = prices[lookback_index]
        future = prices[future_index]
        if current == 0 or previous == 0 or future == 0:
            continue
        feature = current / previous - 1.0
        target = future / current - 1.0
        if feature == 0 or target == 0:
            continue
        pairs.append((epoch, feature, target))

    train = [row for row in pairs if row[0] < split_epoch]
    holdout = [row for row in pairs if row[0] >= split_epoch]
    magnitudes = sorted(abs(feature) for _, feature, _ in train)
    if not magnitudes:
        raise RuntimeError("training data contains no usable momentum rows")
    index = min(len(magnitudes) - 1, int(args.quantile * (len(magnitudes) - 1)))
    threshold = magnitudes[index]

    baseline_correct = sum((feature > 0) == (target > 0) for _, feature, target in holdout)
    threshold_rows = [row for row in holdout if abs(row[1]) >= threshold]
    threshold_correct = sum((feature > 0) == (target > 0) for _, feature, target in threshold_rows)

    baseline_low, baseline_high = _wilson(baseline_correct, len(holdout))
    threshold_low, threshold_high = _wilson(threshold_correct, len(threshold_rows))
    baseline_accuracy = baseline_correct / len(holdout) if holdout else 0.0
    threshold_accuracy = threshold_correct / len(threshold_rows) if threshold_rows else 0.0
    decision_rate = len(threshold_rows) / len(holdout) if holdout else 0.0

    print(f"symbol | {args.symbol}")
    print(f"horizon | {args.horizon} | lookback | {args.lookback} | quantile | {args.quantile:.2f}")
    print(f"train_rows | {len(train)} | holdout_rows | {len(holdout)}")
    print(f"frozen_threshold | {threshold:.12g}")
    print(f"baseline | accuracy | {baseline_accuracy:.4f} | ci95 | {baseline_low:.4f}-{baseline_high:.4f}")
    print(
        f"threshold | accuracy | {threshold_accuracy:.4f} | ci95 | {threshold_low:.4f}-{threshold_high:.4f} | "
        f"decision_rate | {decision_rate:.4f} | decisions | {len(threshold_rows)}"
    )
    print(f"threshold_minus_baseline | {threshold_accuracy - baseline_accuracy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
