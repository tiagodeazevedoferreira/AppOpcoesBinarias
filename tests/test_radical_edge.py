from app_opcoes_binarias.research.dataset import ResearchRow, build_dataset
from app_opcoes_binarias.research.radical_edge import (
    _wilson_lower,
    evaluate_radical,
    fit_radical_index,
    predict_radical,
)


def test_wilson_lower_is_below_observed_rate():
    assert 0.90 < _wilson_lower(99, 100) < 0.99
    assert _wilson_lower(100, 100) < 1.0


def test_radical_engine_is_selective_when_evidence_is_weak():
    rows = [ResearchRow(i, 100.0 + (i % 2), None, None, None, "RISE", 60, None, None) for i in range(80)]
    train = rows[:60]
    test = rows[60:]
    metrics = evaluate_radical(train, test, target_accuracy=0.99, min_evidence=100)
    assert metrics.total_decisions == 0
    assert metrics.decision_rate == 0.0


def test_radical_prediction_never_uses_test_labels():
    ticks = [{"epoch": i, "quote": 100.0 + i * 0.01} for i in range(140)]
    rows = build_dataset(ticks, horizon_seconds=60)
    train, test = rows[:80], rows[80:]
    index = fit_radical_index(train)
    prices = [row.quote for row in train + test]
    prediction = predict_radical(index, prices, len(train), target_accuracy=0.99, min_evidence=2)
    assert prediction.source in {
        "insufficient_history",
        "insufficient_evidence",
        "no_consensus",
        "conflicting_evidence",
        "state+trend",
        "state+motif",
        "trend+motif",
        "state+trend+motif",
    }
