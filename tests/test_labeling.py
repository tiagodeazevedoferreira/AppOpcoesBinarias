from app_opcoes_binarias.research.labeling import (
    PricePoint,
    build_outcome,
    label_direction,
)


def test_direction_labels_are_explicit():
    assert label_direction(100.0, 101.0) == "RISE"
    assert label_direction(100.0, 99.0) == "FALL"
    assert label_direction(100.0, 100.0) == "FLAT"


def test_build_outcome_uses_future_only():
    points = [
        PricePoint(0, 100.0),
        PricePoint(30, 100.2),
        PricePoint(60, 100.5),
    ]
    outcome = build_outcome(points, 0, horizon_seconds=60)
    assert outcome.future_epoch == 60
    assert outcome.future_quote == 100.5
    assert outcome.direction == "RISE"


def test_missing_future_point_is_unlabeled():
    points = [PricePoint(0, 100.0), PricePoint(30, 100.2)]
    outcome = build_outcome(points, 0, horizon_seconds=60)
    assert outcome.future_epoch is None
    assert outcome.direction is None
