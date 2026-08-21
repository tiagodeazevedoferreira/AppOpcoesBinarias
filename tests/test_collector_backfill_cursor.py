from app_opcoes_binarias.data.collector import collect_history_backfill


class FakeClient:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.calls = []

    def get_ticks_history(self, symbol, count=1000, *, start=None, end="latest"):
        self.calls.append((symbol, count, start, end))
        return next(self.batches)


def history(times):
    return {"history": {"times": times, "prices": [100.0 + epoch * 0.01 for epoch in times]}}


def test_backfill_passes_fixed_start_window_and_moves_backwards():
    client = FakeClient([
        history([90, 91, 92]),
        history([87, 88, 89]),
        history([84, 85, 86]),
        history([81, 82, 83]),
        history([80]),
    ])

    result = collect_history_backfill(
        client,
        "frxEURUSD",
        start=80,
        end=92,
        batch_size=3,
        max_batches=5,
    )

    assert [row["epoch"] for row in result] == list(range(80, 93))
    assert client.calls == [
        ("frxEURUSD", 3, 80, 92),
        ("frxEURUSD", 3, 80, 89),
        ("frxEURUSD", 3, 80, 86),
        ("frxEURUSD", 3, 80, 83),
        ("frxEURUSD", 3, 80, 80),
    ]
