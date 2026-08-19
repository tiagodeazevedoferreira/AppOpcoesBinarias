from app_opcoes_binarias.research.baselines import accuracy, majority_class, persistence_prediction


def test_majority_class_uses_training_labels_only():
    assert majority_class(["RISE", "RISE", "FALL"]) == "RISE"


def test_majority_class_has_deterministic_tie_break():
    assert majority_class(["RISE", "FALL"]) == "FALL"


def test_persistence_prediction():
    assert persistence_prediction("RISE") == "RISE"
    assert persistence_prediction(None) is None


def test_accuracy():
    metrics = accuracy(["RISE", "FALL", "RISE"], ["RISE", "RISE", "RISE"])
    assert metrics.total == 3
    assert metrics.correct == 2
    assert metrics.accuracy == 2 / 3
