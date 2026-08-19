from app_opcoes_binarias.research.dataset import ResearchRow, build_dataset, temporal_split


def test_build_dataset_orders_ticks_and_keeps_future_only_in_label():
    rows = build_dataset(
        [
            {"epoch": 120, "quote": 102.0},
            {"epoch": 60, "quote": 101.0},
            {"epoch": 0, "quote": 100.0},
        ]
    )
    assert [row.epoch for row in rows] == [0, 60, 120]
    assert rows[0].return_1 is None
    assert rows[0].ema_distance_10 is None
    assert rows[0].directional_consistency_5 is None
    assert rows[0].label == "RISE"
    assert rows[0].actual_horizon_seconds == 60
    assert rows[2].label is None
    assert rows[2].actual_horizon_seconds is None


def test_temporal_split_does_not_shuffle():
    rows = [
        ResearchRow(i, 100.0, None, None, None, None, None, None, None)
        for i in range(10)
    ]
    train, test = temporal_split(rows, 0.7)
    assert [row.epoch for row in train] == list(range(7))
    assert [row.epoch for row in test] == list(range(7, 10))


def test_dataset_records_actual_horizon_when_ticks_are_irregular():
    rows = build_dataset(
        [
            {"epoch": 0, "quote": 100.0},
            {"epoch": 10, "quote": 100.1},
            {"epoch": 65, "quote": 100.2},
        ]
    )
    assert rows[0].label == "RISE"
    assert rows[0].actual_horizon_seconds == 65


def test_dataset_features_use_only_available_history():
    rows = build_dataset(
        [{"epoch": i, "quote": 100.0 + i} for i in range(12)],
    )
    assert rows[8].ema_distance_10 is None
    assert rows[9].ema_distance_10 is not None
    assert rows[3].directional_consistency_5 is None
    assert rows[4].directional_consistency_5 == 1.0
