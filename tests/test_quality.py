from app_opcoes_binarias.data.quality import assess_ticks


def test_assess_ticks_reports_order_and_uniqueness() -> None:
    ticks = [
        {"symbol": "frxEURUSD", "epoch": 2, "quote": 1.1},
        {"symbol": "frxEURUSD", "epoch": 3, "quote": 1.2},
    ]

    result = assess_ticks(ticks)

    assert result["record_count"] == 2
    assert result["ordered"] is True
    assert result["unique_epochs"] is True
    assert result["valid_shape"] is True
    assert result["first_epoch"] == 2
    assert result["last_epoch"] == 3


def test_assess_ticks_detects_duplicates_and_out_of_order_data() -> None:
    ticks = [
        {"symbol": "frxEURUSD", "epoch": 3, "quote": 1.2},
        {"symbol": "frxEURUSD", "epoch": 3, "quote": 1.3},
        {"symbol": "frxEURUSD", "epoch": 2, "quote": 1.1},
    ]

    result = assess_ticks(ticks)

    assert result["ordered"] is False
    assert result["unique_epochs"] is False


def test_assess_ticks_handles_empty_batch() -> None:
    result = assess_ticks([])

    assert result["record_count"] == 0
    assert result["first_epoch"] is None
    assert result["last_epoch"] is None
