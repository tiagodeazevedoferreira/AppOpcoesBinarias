import json
from typing import Any

import websocket


class DerivPublicClient:
    """Minimal client for Deriv's public market-data WebSocket API."""

    def __init__(self, ws_url: str) -> None:
        self.ws_url = ws_url
        self._ws: websocket.WebSocket | None = None

    def connect(self) -> None:
        self._ws = websocket.create_connection(self.ws_url, timeout=15)

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("Deriv WebSocket is not connected")

        self._ws.send(json.dumps(payload))
        raw = self._ws.recv()
        response = json.loads(raw)

        if "error" in response:
            error = response["error"]
            raise RuntimeError(f"Deriv API error {error.get('code')}: {error.get('message')}")

        return response

    def get_active_symbols(self) -> dict[str, Any]:
        return self.request({"active_symbols": "brief", "product_type": "basic"})

    def get_ticks_history(self, symbol: str, count: int = 1000) -> dict[str, Any]:
        if count < 1:
            raise ValueError("count must be greater than zero")
        return self.request(
            {
                "ticks_history": symbol,
                "count": count,
                "end": "latest",
                "style": "ticks",
            }
        )

    def get_ticks(self, symbol: str) -> dict[str, Any]:
        return self.request({"ticks": symbol, "subscribe": 1})
