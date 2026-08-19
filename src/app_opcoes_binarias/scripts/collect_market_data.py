import argparse
import json
import logging
from pathlib import Path

from app_opcoes_binarias.config.settings import settings
from app_opcoes_binarias.data.deriv_client import DerivPublicClient

logger = logging.getLogger(__name__)


def collect(symbol: str, count: int, output: Path) -> int:
    client = DerivPublicClient(settings.deriv_ws_url)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        client.connect()
        history = client.get_ticks_history(symbol, count=count)
        prices = history.get("history", {}).get("prices", [])
        times = history.get("history", {}).get("times", [])
        if len(prices) != len(times):
            raise ValueError("Deriv history returned mismatched prices and times")
        with output.open("w", encoding="utf-8") as handle:
            for epoch, quote in zip(times, prices, strict=True):
                handle.write(json.dumps({"symbol": symbol, "epoch": epoch, "quote": str(quote)}) + "\n")
                written += 1
    finally:
        client.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a bounded Deriv market-data sample")
    parser.add_argument("--symbol", default=settings.market_symbol)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("data/raw/eurusd_ticks.jsonl"))
    args = parser.parse_args()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
    written = collect(args.symbol, args.count, args.output)
    logger.info("Collected %s historical ticks for %s", written, args.symbol)


if __name__ == "__main__":
    main()
