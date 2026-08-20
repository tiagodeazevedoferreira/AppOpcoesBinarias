from __future__ import annotations

import bisect

from .baselines import ClassificationMetrics, accuracy
from .dataset import ResearchRow


def evaluate_momentum_baseline(
    train: list[ResearchRow],
    test: list[ResearchRow],
    lookback_seconds: int = 60,
) -> ClassificationMetrics:
    """Evaluate a leakage-safe observable momentum baseline.

    The prediction at each test row uses only the quote history available at the
    observation epoch: direction of the return over the requested lookback.
    """
    if lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be positive")

    combined = sorted(train + test, key=lambda row: row.epoch)
    if not combined:
        return ClassificationMetrics(accuracy=0.0, total=0, correct=0)

    epochs = [row.epoch for row in combined]
    quotes = [row.quote for row in combined]
    by_epoch = {row.epoch: row for row in test if row.label is not None}

    y_true: list[str] = []
    y_pred: list[str | None] = []
    for row in test:
        if row.label is None or row.epoch not in by_epoch:
            continue
        idx = bisect.bisect_right(epochs, row.epoch - lookback_seconds) - 1
        if idx < 0:
            continue
        previous_quote = quotes[idx]
        if previous_quote == 0 or row.quote == 0:
            continue
        momentum = row.quote / previous_quote - 1.0
        if momentum == 0:
            continue
        prediction = "RISE" if momentum > 0 else "FALL"
        y_true.append(row.label)
        y_pred.append(prediction)

    return accuracy(y_true, y_pred)
