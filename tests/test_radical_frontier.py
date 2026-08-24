from app_opcoes_binarias.research.dataset import ResearchRow
from app_opcoes_binarias.research.radical_frontier import evaluate_frontier


def _rows(count: int) -> list[ResearchRow]:
    return [
        ResearchRow(
            epoch=i,
            quote=100.0 + (i * 0.01),
            return_1=0.0001,
            momentum_2=0.0002,
            volatility_5=0.00001,
            label="RISE",
            actual_horizon_seconds=60,
            ema_distance_10=0.0001,
            directional_consistency_5=1.0,
        )
        for i in range(count)
    ]


def test_frontier_is_leakage_safe_and_reports_points():
    rows = _rows(180)
    report = evaluate_frontier(
        rows[:120],
        rows[120:],
        thresholds=(0.80,),
        evidence_levels=(2,),
        min_views_levels=(1,),
    )
    assert report.points
    assert report.best_accuracy is not None
    assert report.best_accuracy.accuracy == 1.0
    assert report.best_accuracy.decisions > 0
