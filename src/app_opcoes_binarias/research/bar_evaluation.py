from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class BarEvaluation:
    bar_seconds: int
    rows: int
    train_rows: int
    test_rows: int
    flat_ratio: float
    persistence_accuracy: float
    balanced_accuracy: float
    non_overlapping_persistence_accuracy: float
    non_overlapping_balanced_accuracy: float
    non_overlapping_rows: int


def _bars(ticks: list[dict[str, object]], bar_seconds: int) -> list[tuple[int, float]]:
    grouped: dict[int, tuple[int, float]] = {}
    for item in sorted(ticks, key=lambda value: int(value["epoch"])):
        epoch = int(item["epoch"])
        quote = float(item["quote"])
        bucket = (epoch // bar_seconds) * bar_seconds
        grouped[bucket] = (epoch, quote)
    return sorted(grouped.values())


def _labels(bars: list[tuple[int, float]], horizon_seconds: int) -> list[str | None]:
    epochs = [epoch for epoch, _ in bars]
    prices = [price for _, price in bars]
    labels: list[str | None] = []
    for index, epoch in enumerate(epochs):
        future = bisect_left(epochs, epoch + horizon_seconds, index + 1)
        if future >= len(epochs):
            labels.append(None)
            continue
        delta = prices[future] / prices[index] - 1.0 if prices[index] else 0.0
        labels.append("RISE" if delta > 0 else "FALL" if delta < 0 else "FLAT")
    return labels


def _metrics(train: list[str], test: list[str]) -> tuple[float, float]:
    if not test:
        return 0.0, 0.0
    previous = train[-1] if train else None
    counts = Counter(test)
    correct = Counter()
    total_correct = 0
    for label in test:
        if previous == label:
            correct[label] += 1
            total_correct += 1
        previous = label
    recalls = [correct[label] / counts[label] if counts[label] else 0.0 for label in ("RISE", "FALL", "FLAT")]
    return total_correct / len(test), sum(recalls) / 3.0


def evaluate_bar_sizes(
    ticks: list[dict[str, object]],
    *,
    horizon_seconds: int,
    train_ratio: float,
    bar_sizes: tuple[int, ...],
) -> tuple[BarEvaluation, ...]:
    if horizon_seconds <= 0 or not 0 < train_ratio < 1:
        raise ValueError("invalid horizon or train_ratio")
    if not bar_sizes or any(value <= 0 for value in bar_sizes):
        raise ValueError("bar_sizes must be positive and non-empty")

    reports: list[BarEvaluation] = []
    for bar_seconds in bar_sizes:
        bars = _bars(ticks, bar_seconds)
        labels = _labels(bars, horizon_seconds)
        usable = [label for label in labels if label is not None]
        cut = int(len(labels) * train_ratio)
        train = [label for label in labels[:cut] if label is not None]
        test = [label for label in labels[cut:] if label is not None]
        persistence, balanced = _metrics(train, test)

        selected: list[str] = []
        last_epoch: int | None = None
        for index, (epoch, _) in enumerate(bars):
            label = labels[index]
            if label is None:
                continue
            if last_epoch is None or epoch >= last_epoch + horizon_seconds:
                selected.append(label)
                last_epoch = epoch
        non_cut = int(len(selected) * train_ratio)
        non_persistence, non_balanced = _metrics(selected[:non_cut], selected[non_cut:])

        reports.append(
            BarEvaluation(
                bar_seconds=bar_seconds,
                rows=len(usable),
                train_rows=len(train),
                test_rows=len(test),
                flat_ratio=sum(label == "FLAT" for label in usable) / len(usable) if usable else 0.0,
                persistence_accuracy=persistence,
                balanced_accuracy=balanced,
                non_overlapping_persistence_accuracy=non_persistence,
                non_overlapping_balanced_accuracy=non_balanced,
                non_overlapping_rows=len(selected),
            )
        )
    return tuple(reports)
