from app_opcoes_binarias.research.dataset import ResearchRow
from app_opcoes_binarias.research.evaluation import evaluate_baselines


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
