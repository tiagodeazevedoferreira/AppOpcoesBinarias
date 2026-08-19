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
    assert rows[0].label == "UP"
    assert rows[2].label is None


def test_temporal_split_does_not_shuffle():
    rows = [ResearchRow(i, 100.0, None, None, None, None) for i in range(10)]
    train, test = temporal_split(rows, 0.7)
    assert [row.epoch for row in train] == list(range(7))
    assert [row.epoch for row in test] == list(range(7, 10))
