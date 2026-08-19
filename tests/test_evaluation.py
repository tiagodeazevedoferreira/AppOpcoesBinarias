from app_opcoes_binarias.research.dataset import ResearchRow
from app_opcoes_binarias.research.evaluation import evaluate_baselines, sample_non_overlapping


def row(epoch: int, label: str | None) -> ResearchRow:
    return ResearchRow(epoch, 100.0, None, None, None, label, None)


def test_baselines_use_train_only_for_majority_and_carry_train_direction_into_test():
    train = [row(1, "RISE"), row(2, "RISE"), row(3, "FALL")]
    test = [row(4, "FALL"), row(5, "FALL"), row(6, "RISE")]

    report = evaluate_baselines(train, test)

    assert report.majority.total == 3
    assert report.majority.correct == 1
    assert report.train_distribution == {"RISE": 2, "FALL": 1}
    assert report.test_distribution == {"FALL": 2, "RISE": 1}
    assert report.persistence.correct == 2


def test_baselines_reject_empty_labeled_training_data():
    try:
        evaluate_baselines([row(1, None)], [row(2, "RISE")])
    except ValueError as exc:
        assert "train" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_non_overlapping_sampling_enforces_horizon_between_decisions():
    rows = [row(0, "RISE"), row(1, "FALL"), row(59, "RISE"), row(60, "FALL"), row(61, "RISE"), row(120, "FALL")]

    selected = sample_non_overlapping(rows, 60)

    assert [item.epoch for item in selected] == [0, 60, 120]


def test_non_overlapping_sampling_rejects_invalid_horizon():
    try:
        sample_non_overlapping([row(1, "RISE")], 0)
    except ValueError as exc:
        assert "horizon" in str(exc)
    else:
        raise AssertionError("expected ValueError")
