from __future__ import annotations

import argparse
import bisect

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.firebase_store import FirebaseStore
from app_opcoes_binarias.data.tick_storage import TickStorage


def _evaluate(ticks: list[dict[str, object]], *, horizon: int, lookback: int, folds: int) -> tuple[float, float, float, int, tuple[float, ...]]:
    ordered = sorted(((int(item["epoch"]), float(item["quote"])) for item in ticks), key=lambda item: item[0])
    epochs = [item[0] for item in ordered]
    prices = [item[1] for item in ordered]
    rows: list[tuple[int, float]] = []
    for i, epoch in enumerate(epochs):
        lb = bisect.bisect_right(epochs, epoch - lookback, 0, i + 1) - 1
        future = bisect.bisect_left(epochs, epoch + horizon, i + 1)
        if lb < 0 or future >= len(epochs) or prices[i] == 0 or prices[lb] == 0:
            continue
        feature = prices[i] / prices[lb] - 1.0
        target = prices[future] / prices[i] - 1.0
        if feature == 0 or target == 0:
            continue
        rows.append((epoch, 1.0 if (feature > 0) == (target > 0) else 0.0))

    selected: list[tuple[int, float]] = []
    last_epoch: int | None = None
    for row in rows:
        if last_epoch is None or row[0] >= last_epoch + horizon:
            selected.append(row)
            last_epoch = row[0]

    if not selected or len(selected) < folds:
        return 0.0, 0.0, 0.0, 0, ()

    fold_size = len(selected) // folds
    accuracies: list[float] = []
    for fold in range(folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < folds - 1 else len(selected)
        test = selected[start:end]
        accuracies.append(sum(value for _, value in test) / len(test))

    mean = sum(accuracies) / len(accuracies)
    return mean, min(accuracies), max(accuracies), sum(value >= 0.5 for value in accuracies), tuple(accuracies)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate signal stability at fixed lookback/horizon ratios.")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--pairs", default="30:15,60:30,120:60,300:150")
    parser.add_argument("--folds", type=int, default=10)
    args = parser.parse_args()

    if not settings.firebase_database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required for research")
    pairs = [(int(h), int(l)) for h, l in (part.split(":") for part in args.pairs.split(",") if part.strip())]
    ticks = TickStorage(FirebaseStore(settings.firebase_database_url)).read_all(args.symbol)

    print("horizon | lookback | ratio | mean | min | max | folds>=chance | folds")
    for horizon, lookback in pairs:
        if lookback >= horizon or lookback <= 0:
            print(f"{horizon:>7} | {lookback:>8} | INVALID: lookback must be < horizon")
            continue
        mean, minimum, maximum, at_chance, folds = _evaluate(ticks, horizon=horizon, lookback=lookback, folds=args.folds)
        ratio = lookback / horizon
        fold_text = ",".join(f"{value:.4f}" for value in folds)
        print(f"{horizon:>7} | {lookback:>8} | {ratio:.2f} | {mean:.4f} | {minimum:.4f} | {maximum:.4f} | {at_chance:>14} | {fold_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
