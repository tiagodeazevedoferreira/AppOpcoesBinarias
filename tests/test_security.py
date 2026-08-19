from pathlib import Path


def test_no_secret_files_are_tracked_by_default() -> None:
    assert not Path(".env").exists()
    assert not list(Path(".").glob("*firebase*.json"))
