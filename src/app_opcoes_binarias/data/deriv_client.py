import json
import logging
import time
from collections.abc import Iterator
from typing import Any

import websocket

from app_opcoes_binarias.data.models import MarketTick

logger = logging.getLogger(__name__)


class DerivPublicClient:
    """Client for Deriv's current public market-data WebSocket."""

    def __init__(self, ws_url: str, timeout: float = 15.0) -> None:
        self.ws_url = ws_url
        self.timeout = timeout
        self._ws: websocket.WebSocket | None = None

    def connect(self) -> None:
        self.close()
        self._ws = websocket.create_connection(self.ws_url, timeout=self.timeout)
        logger.info("Connected to Deriv public market-data WebSocket")

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("Deriv WebSocket is not connected")
        self._ws.send(json.dumps(payload, separators=(",", ":")))
        raw = self._ws.recv()
        if raw is None:
            raise ConnectionError("Deriv WebSocket returned no data")
        response = json.loads(raw)
        if not isinstance(response, dict):
            raise TypeError("Deriv response is not a JSON object")
        if "error" in response:
            error = response["error"]
            code = error.get("code", "UNKNOWN") if isinstance(error, dict) else "UNKNOWN"
            message = error.get("message", "Unknown error") if isinstance(error, dict) else str(error)
            raise RuntimeError(f"Deriv API error {code}: {message}")
        return response

    def get_active_symbols(self, contract_type: list[str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"active_symbols": "brief", "req_id": 1}
        if contract_type:
            payload["contract_type"] = contract_type
        return self.request(payload)

    def get_contracts_for(self, symbol: str) -> dict[str, Any]:
        if not symbol:
            raise ValueError("symbol is required")
        return self.request({"contracts_for": symbol, "req_id": 2})

    def get_ticks_history(
        self,
        symbol: str,
        count: int = 1000,
        *,
        start: int | None = None,
        end: int | str = "latest",
    ) -> dict[str, Any]:
        if not symbol:
            raise ValueError("symbol is required")
        if count < 1:
            raise ValueError("count must be greater than zero")
        if start is not None and start < 0:
            raise ValueError("start must be non-negative")
        if isinstance(end, int) and end < 0:
            raise ValueError("end must be non-negative")
        payload: dict[str, Any] = {
            "ticks_history": symbol,
            "count": count,
            "end": end,
            "style": "ticks",
            "req_id": 3,
        }
        if start is not None:
            payload["start"] = start
        return self.request(payload)

    def subscribe_ticks(self, symbol: str) -> None:
        if not symbol:
            raise ValueError("symbol is required")
        if self._ws is None:
            raise RuntimeError("Deriv WebSocket is not connected")
        self._ws.send(json.dumps({"ticks": symbol, "subscribe": 1, "req_id": 4}))

    def tick_stream(self, symbol: str, max_ticks: int | None = None) -> Iterator[MarketTick]:
        self.subscribe_ticks(symbol)
        received = 0
        while max_ticks is None or received < max_ticks:
            if self._ws is None:
                raise ConnectionError("Deriv WebSocket is not connected")
            raw = self._ws.recv()
            if raw is None:
                raise ConnectionError("Deriv WebSocket returned no data")
            response = json.loads(raw)
            if not isinstance(response, dict):
                continue
            if response.get("error"):
                error = response["error"]
                message = error.get("message", "Unknown error") if isinstance(error, dict) else str(error)
                raise RuntimeError(f"Deriv stream error: {message}")
            if response.get("msg_type") != "tick":
                continue
            tick = MarketTick.from_deriv(response)
            received += 1
            yield tick

    def collect_with_reconnect(
        self,
        symbol: str,
        max_ticks: int,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> Iterator[MarketTick]:
        """Collect a bounded sample, reconnecting with exponential backoff."""
        if max_ticks < 1:
            raise ValueError("max_ticks must be greater than zero")
        collected = 0
        retries = 0
        while collected < max_ticks:
            try:
                if self._ws is None:
                    self.connect()
                for tick in self.tick_stream(symbol, max_ticks=max_ticks - collected):
                    collected += 1
                    retries = 0
                    yield tick
            except (websocket.WebSocketException, OSError, ConnectionError, TimeoutError) as exc:
                self.close()
                if retries >= max_retries:
                    raise RuntimeError("Maximum WebSocket reconnect attempts exceeded") from exc
                delay = base_delay * (2**retries)
                retries += 1
                logger.warning("Connection lost; retry %s/%s in %.1fs: %s", retries, max_retries, delay, exc)
                time.sleep(delay)
            finally:
                if collected >= max_ticks:
                    self.close()
