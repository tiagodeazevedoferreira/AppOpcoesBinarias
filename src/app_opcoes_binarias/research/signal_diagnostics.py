from __future__ import annotations

import bisect
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SignalDiagnostic:
    lookback_seconds: int
    correlation: float
    train_sign_accuracy: float
    test_sign_accuracy: float
    non_overlapping_sign_accuracy: float
    train_rows: int
    test_rows: int
    non_overlapping_rows: int


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


def _sign_accuracy(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return 0.0
    usable = sum(x != 0 and y != 0 for x, y in zip(xs, ys))
    return (
        sum((x > 0) == (y > 0) for x, y in zip(xs, ys) if x != 0 and y != 0) / usable
        if usable
        else 0.0
    )


def evaluate_signal_diagnostics(
    ticks: list[dict[str, object]],
    *,
    horizon_seconds: int = 60,
    train_ratio: float = 0.7,
    lookbacks: tuple[int, ...] = (1, 2, 5, 10, 30, 60),
) -> tuple[SignalDiagnostic, ...]:
    if horizon_seconds <= 0 or not 0 < train_ratio < 1:
        raise ValueError("invalid horizon or train_ratio")
    ordered = sorted(
        ((int(item["epoch"]), float(item["quote"])) for item in ticks),
        key=lambda item: item[0],
    )
    epochs = [item[0] for item in ordered]
    prices = [item[1] for item in ordered]
    reports: list[SignalDiagnostic] = []
    for lookback in lookbacks:
        xs: list[float] = []
        ys: list[float] = []
        epochs_used: list[int] = []
        for i, epoch in enumerate(epochs):
            lookback_index = bisect.bisect_right(epochs, epoch - lookback, 0, i + 1) - 1
            if lookback_index < 0 or prices[lookback_index] == 0 or prices[i] == 0:
                continue
            future = bisect.bisect_left(epochs, epoch + horizon_seconds, i + 1)
            if future >= len(epochs):
                continue
            feature = prices[i] / prices[lookback_index] - 1.0
            target = prices[future] / prices[i] - 1.0
            xs.append(feature)
            ys.append(target)
            epochs_used.append(epoch)
        cut = int(len(xs) * train_ratio)
        train_x, train_y = xs[:cut], ys[:cut]
        test_x, test_y = xs[cut:], ys[cut:]
        selected_indices: list[int] = []
        last_epoch: int | None = None
        for index, epoch in enumerate(epochs_used):
            if last_epoch is None or epoch >= last_epoch + horizon_seconds:
                selected_indices.append(index)
                last_epoch = epoch
        non_cut = int(len(selected_indices) * train_ratio)
        non_test_indices = selected_indices[non_cut:]
        non_x = [xs[i] for i in non_test_indices]
        non_y = [ys[i] for i in non_test_indices]
        reports.append(
            SignalDiagnostic(
                lookback_seconds=lookback,
                correlation=_corr(xs, ys),
                train_sign_accuracy=_sign_accuracy(train_x, train_y),
                test_sign_accuracy=_sign_accuracy(test_x, test_y),
                non_overlapping_sign_accuracy=_sign_accuracy(non_x, non_y),
                train_rows=len(train_x),
                test_rows=len(test_x),
                non_overlapping_rows=len(non_x),
            )
        )
    return tuple(reports)
