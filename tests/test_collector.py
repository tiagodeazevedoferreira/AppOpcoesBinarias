from app_opcoes_binarias.data.collector import normalize_tick


def test_normalize_tick() -> None:
    response = {
        "tick": {
            "symbol": "frxEURUSD",
            "epoch": 1766140800,
            "quote": 1.17345,
        }
    }

    tick = normalize_tick(response)

    assert tick["symbol"] == "frxEURUSD"
    assert tick["epoch"] == 1766140800
    assert tick["quote"] == 1.17345
    assert tick["timestamp"].endswith("+00:00")
