import pytest

from app_opcoes_binarias.research.label_tolerance_evaluation import evaluate_tolerance_sweep


def _ticks() -> list[dict[str, object]]:
    return [{"epoch": i * 60, "quote": 100.0 + (0.001 if i % 2 else 0.0)} for i in range(20)]


def test_tolerance_sweep_returns_sorted_requested_reports() -> None:
    reports = evaluate_tolerance_sweep(
        _ticks(),
        horizon_seconds=60,
        train_ratio=0.7,
        tolerances=(0.001, 0.0),
    )

    assert [report.tolerance for report in reports] == [0.001, 0.0]
    assert all(0.0 <= report.flat_ratio <= 1.0 for report in reports)


def test_tolerance_sweep_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_tolerance_sweep(
            _ticks(),
            horizon_seconds=60,
            train_ratio=0.7,
            tolerances=(-0.001,),
        )
