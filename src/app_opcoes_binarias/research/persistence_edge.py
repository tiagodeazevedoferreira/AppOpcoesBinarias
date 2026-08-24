"""Observable persistence diagnostics for selective edge discovery.

The old label-persistence baseline is intentionally excluded here because it
uses the previous row's *future* label. This module only uses information
available at the observation timestamp: the sign of the return over a
lookback window, plus coarse observable market-state descriptors.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .dataset import ResearchRow


@dataclass(frozen=True)
class PersistenceCell:
    key: tuple[int, ...]
    prediction: str
    evidence: int
    correct: int
    accuracy: float
    lower_bound: float


@dataclass(frozen=True)
class PersistenceDiagnostic:
    horizon_seconds: int
    lookback_seconds: int
    train_rows: int
    test_rows: int
    eligible_test_rows: int
    cells: tuple[PersistenceCell, ...]
    top_cells: tuple[PersistenceCell, ...]


def _wilson_lower(correct: int, total: int, z: float = 2.326347874) -> float:
    if total <= 0:
        return 0.0
    p = correct / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    spread = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return max(0.0, (center - spread) / denominator)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _bucket(value: float, cuts: Sequence[float]) -> int:
    for index, cut in enumerate(cuts):
        if value <= cut:
            return index
    return len(cuts)


def _observable_key(rows: Sequence[ResearchRow], position: int, lookback: int) -> tuple[int, ...] | None:
    if position < lookback:
        return None
    current = rows[position].quote
    previous = rows[position - lookback].quote
    if previous == 0:
        return None

    returns = [
        rows[position].quote / rows[position - lag].quote - 1.0
        for lag in (lookback // 4, lookback // 2, lookback)
        if lag > 0 and position >= lag and rows[position - lag].quote != 0
    ]
    if len(returns) != 3:
        return None

    start = position - lookback
    changes = [
        _sign(rows[p].quote - rows[p - 1].quote)
        for p in range(start + 1, position + 1)
    ]
    non_zero = [change for change in changes if change]
    consistency = (
        max(sum(change > 0 for change in non_zero), sum(change < 0 for change in non_zero))
        / len(non_zero)
        if non_zero
        else 0.0
    )
    path = sum(abs(rows[p].quote - rows[p - 1].quote) for p in range(start + 1, position + 1))
    displacement = abs(current - previous)
    efficiency = displacement / path if path else 0.0

    recent_returns = [
        rows[p].quote / rows[p - 1].quote - 1.0
        for p in range(max(1, position - 29), position + 1)
        if rows[p - 1].quote != 0
    ]
    if len(recent_returns) < 2:
        volatility = 0.0
    else:
        mean = sum(recent_returns) / len(recent_returns)
        volatility = math.sqrt(sum((value - mean) ** 2 for value in recent_returns) / len(recent_returns))

    direction = _sign(returns[-1])
    return (
        direction,
        _sign(returns[0]),
        _sign(returns[1]),
        _bucket(consistency, (0.55, 0.65, 0.75, 0.85, 0.92, 0.97)),
        _bucket(efficiency, (0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85)),
        _bucket(math.log10(max(volatility, 1e-12)), (-5.0, -4.5, -4.0, -3.5, -3.0, -2.5)),
        int(rows[position].epoch // 3600) % 24,
    )


def diagnose_persistence_edge(
    train: Sequence[ResearchRow],
    test: Sequence[ResearchRow],
    *,
    horizon_seconds: int,
    lookback_seconds: int,
    min_evidence: int = 20,
    top_n: int = 30,
) -> PersistenceDiagnostic:
    if horizon_seconds <= 0 or lookback_seconds <= 0:
        raise ValueError("horizon_seconds and lookback_seconds must be positive")
    if min_evidence < 2:
        raise ValueError("min_evidence must be at least 2")

    train_rows = sorted(train, key=lambda row: row.epoch)
    test_rows = sorted(test, key=lambda row: row.epoch)
    combined = train_rows + test_rows
    train_count = len(train_rows)

    evidence: dict[tuple[tuple[int, ...], str], list[int]] = defaultdict(lambda: [0, 0])
    for position, row in enumerate(train_rows):
        key = _observable_key(combined[:train_count], position, lookback_seconds)
        if key is None or row.label not in {"RISE", "FALL"}:
            continue
        prediction = "RISE" if key[0] > 0 else "FALL" if key[0] < 0 else None
        if prediction is None:
            continue
        cell = evidence[(key, prediction)]
        cell[0] += 1
        cell[1] += int(prediction == row.label)

    cells: list[PersistenceCell] = []
    for (key, prediction), (total, correct) in evidence.items():
        if total < min_evidence:
            continue
        cells.append(
            PersistenceCell(
                key=key,
                prediction=prediction,
                evidence=total,
                correct=correct,
                accuracy=correct / total,
                lower_bound=_wilson_lower(correct, total),
            )
        )

    cells.sort(key=lambda cell: (cell.lower_bound, cell.accuracy, cell.evidence), reverse=True)

    eligible = 0
    for offset, row in enumerate(test_rows):
        if row.label not in {"RISE", "FALL"}:
            continue
        key = _observable_key(combined, train_count + offset, lookback_seconds)
        if key is not None and key[0] != 0:
            eligible += 1

    return PersistenceDiagnostic(
        horizon_seconds=horizon_seconds,
        lookback_seconds=lookback_seconds,
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        eligible_test_rows=eligible,
        cells=tuple(cells),
        top_cells=tuple(cells[:top_n]),
    )
