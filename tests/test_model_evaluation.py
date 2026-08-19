from app_opcoes_binarias.research.dataset import ResearchRow
from app_opcoes_binarias.research.model_evaluation import evaluate_nearest_centroid


def row(epoch: int, label: str | None, value: float) -> ResearchRow:
    return ResearchRow(
        epoch,
        value,
        value - 100.0,
        value - 100.0,
        0.01,
        label,
        60.0,
        value - 100.0,
        0.8,
    )


def test_evaluation_fits_only_on_train_and_reports_test_accuracy() -> None:
    report = evaluate_nearest_centroid(
        [row(1, "RISE", 101.0), row(2, "FALL", 99.0)],
        [row(3, "RISE", 101.5), row(4, "FALL", 98.5)],
    )
    assert report.usable_test_rows == 2
    assert report.skipped_test_rows == 0
    assert report.model.total == 2


def test_evaluation_skips_incomplete_test_rows() -> None:
    report = evaluate_nearest_centroid(
        [row(1, "RISE", 101.0), row(2, "FALL", 99.0)],
        [ResearchRow(3, 100.0, None, None, None, "RISE", 60.0), row(4, "FALL", 98.5)],
    )
    assert report.usable_test_rows == 1
    assert report.skipped_test_rows == 1
