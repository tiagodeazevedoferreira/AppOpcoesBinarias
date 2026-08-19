from app_opcoes_binarias.research.baselines import ClassificationMetrics
from app_opcoes_binarias.research.stability import evaluate_walk_forward_stability
from app_opcoes_binarias.research.walk_forward import FoldReport, WalkForwardReport


def metrics(accuracy: float, total: int = 100) -> ClassificationMetrics:
    return ClassificationMetrics(accuracy=accuracy, correct=round(accuracy * total), total=total)


def test_stability_counts_consistent_folds() -> None:
    folds = (
        FoldReport(100, 100, metrics(0.60), metrics(0.55), metrics(0.52), metrics(0.58)),
        FoldReport(200, 100, metrics(0.40), metrics(0.52), metrics(0.48), metrics(0.49)),
    )
    report = evaluate_walk_forward_stability(
        WalkForwardReport(
            folds=folds,
            majority=metrics(0.50, 200),
            persistence=metrics(0.535, 200),
            nearest_centroid=metrics(0.50, 200),
            softmax=metrics(0.535, 200),
        )
    )

    assert report.softmax.folds == 2
    assert report.softmax.folds_at_or_above_chance == 1
    assert report.softmax.folds_better_than_persistence == 1
    assert report.softmax.chance_gap == 0.03500000000000003
    assert report.softmax.naive_ci95_low < report.softmax.accuracy < report.softmax.naive_ci95_high
