from app_opcoes_binarias.research.signal_walk_forward import evaluate_signal_walk_forward


def test_signal_walk_forward_returns_requested_folds() -> None:
    ticks = [{"epoch": i, "quote": 100.0 + (0.1 if i % 2 else 0.0)} for i in range(500)]
    reports = evaluate_signal_walk_forward(ticks, horizon_seconds=10, folds=5, lookbacks=(1,))
    assert len(reports) == 1
    assert len(reports[0].folds) == 5
    assert all(fold.rows >= 0 for fold in reports[0].folds)


def test_signal_walk_forward_rejects_invalid_folds() -> None:
    try:
        evaluate_signal_walk_forward([], folds=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
