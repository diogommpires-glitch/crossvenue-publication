"""Read-only Binance public market-data connector.

This connector can read public depth data. Authenticated account access and
exchange order submission are intentionally disabled; use the local paper
account for all simulated execution.
"""

import json
import time
import threading
from typing import Dict, Optional
import websocket
import requests
from queue import Queue, Empty

from .base_connector import (
    BaseConnector,
    ConnectionStatus,
    OrderBookSnapshot,
    OrderResponse,
)
from ..utils.types import Side, Price, Quantity, Timestamp
from ..orderbook.order import Fill


class BinanceConnector(BaseConnector):
    """Read-only public Binance market data; order and account routes are disabled."""

    # Binance API endpoints
    REST_BASE = "https://api.binance.com"
    REST_TESTNET = "https://testnet.binance.vision"
    WS_BASE = "wss://stream.binance.com:9443"
    WS_TESTNET = "wss://testnet.binance.vision"

    def __init__(self, symbol: str, config: Optional[Dict] = None):
        """Initialize Binance connector.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            config: Public-data settings such as testnet and rate limit; credentials are ignored.
        """
        super().__init__(symbol, config)

        # Normalize None config to empty dict
        if config is None:
            config = {}

        # API credentials
        # Never retain credentials: this connector is public-data only.
        self.api_key = ""
        self.api_secret = ""

        # Environment selection
        self.testnet = config.get("testnet", False)
        self.rest_base = self.REST_TESTNET if self.testnet else self.REST_BASE
        self.ws_base = self.WS_TESTNET if self.testnet else self.WS_BASE

        # Rate limiting
        self.rate_limit = config.get("rate_limit_per_minute", 1200)
        self._request_times: Queue = Queue()

        # WebSocket state
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False
        self._reconnect_delay = 1.0  # seconds, exponential backoff
        self._max_reconnect_delay = 60.0

        # Order book state
        self._orderbook_lock = threading.RLock()
        self._current_orderbook: Optional[OrderBookSnapshot] = None
        self._last_update_id = 0

        # Normalize symbol (Binance uses uppercase, no separator)
        self.symbol = symbol.upper().replace("/", "").replace("-", "")

    def connect(self) -> None:
        """Connect to Binance WebSocket and REST API."""
        if self._status == ConnectionStatus.CONNECTED:
            return

        self._status = ConnectionStatus.CONNECTING
        self._running = True

        # Verify REST API access
        try:
            self._verify_rest_connection()
        except Exception as e:
            self._status = ConnectionStatus.ERROR
            raise ConnectionError(f"REST API connection failed: {e}")

        # Start WebSocket connection
        self._start_websocket()

        self._status = ConnectionStatus.CONNECTED
        print(f"[OK] Connected to Binance {self.symbol}")

    def disconnect(self) -> None:
        """Disconnect from Binance."""
        self._running = False

        if self._ws:
            self._ws.close()

        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5.0)

        self._status = ConnectionStatus.DISCONNECTED
        print(f"[OK] Disconnected from Binance {self.symbol}")

    def get_orderbook(self) -> OrderBookSnapshot:
        """Get current order book snapshot.

        Returns:
            Current order book state

        Raises:
            RuntimeError: If not connected or no data available
        """
        if self._status != ConnectionStatus.CONNECTED:
            raise RuntimeError("Not connected to exchange")

        with self._orderbook_lock:
            if self._current_orderbook is None:
                raise RuntimeError("No order book data available yet")
            return self._current_orderbook

    def submit_market_order(self, side: Side, quantity: Quantity, client_order_id: Optional[str] = None) -> OrderResponse:
        """Reject exchange execution; all trading in this project is paper-only."""
        raise NotImplementedError("Binance execution is disabled; use the local paper-trading account")


    def submit_limit_order(self, side: Side, price: Price, quantity: Quantity, client_order_id: Optional[str] = None) -> OrderResponse:
        """Reject exchange execution; all trading in this project is paper-only."""
        raise NotImplementedError("Binance execution is disabled; use the local paper-trading account")


    def cancel_order(self, order_id: str) -> OrderResponse:
        """Reject exchange cancellation; this class does not create exchange orders."""
        raise NotImplementedError("Binance order management is disabled; no exchange orders are created")


    def get_balance(self, asset: str) -> float:
        """Reject authenticated account queries; use the local paper ledger."""
        raise NotImplementedError("Authenticated Binance account access is disabled")


    # Private methods

    def _verify_rest_connection(self) -> None:
        """Check the public REST endpoint only; never make authenticated requests."""
        response = requests.get(f"{self.rest_base}/api/v3/ping", timeout=10)
        response.raise_for_status()


    def _start_websocket(self) -> None:
        """Start WebSocket connection for order book."""
        # Binance depth stream: 100ms updates, 20 levels
        stream = f"{self.symbol.lower()}@depth20@100ms"
        ws_url = f"{self.ws_base}/ws/{stream}"

        self._ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_ws_open,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close
        )

        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()

        # Wait for initial connection
        time.sleep(1.0)

    def _on_ws_open(self, ws) -> None:
        """WebSocket connection opened."""
        print(f"[OK] WebSocket connected: {self.symbol}")
        self._reconnect_delay = 1.0  # Reset backoff

    def _on_ws_message(self, ws, message: str) -> None:
        """Process WebSocket message."""
        try:
            data = json.loads(message)

            # Parse order book snapshot
            bids = [(Price(float(p)), Quantity(float(q)))
                    for p, q in data.get("bids", [])]
            asks = [(Price(float(p)), Quantity(float(q)))
                    for p, q in data.get("asks", [])]

            snapshot = OrderBookSnapshot(
                timestamp=Timestamp(time.time()),
                bids=bids,
                asks=asks,
                symbol=self.symbol
            )

            with self._orderbook_lock:
                self._current_orderbook = snapshot
                self._last_update_id = data.get("lastUpdateId", 0)

            # Notify subscribers
            self._notify_orderbook_update(snapshot)

        except Exception as e:
            print(f"[ERR] WebSocket message parse error: {e}")

    def _on_ws_error(self, ws, error) -> None:
        """WebSocket error occurred."""
        print(f"[ERR] WebSocket error: {error}")
        self._status = ConnectionStatus.ERROR

    def _on_ws_close(self, ws, close_status_code, close_msg) -> None:
        """WebSocket connection closed."""
        print(f"[--] WebSocket closed: {close_status_code} {close_msg}")

        # Attempt reconnection if still running
        if self._running:
            print(f"[--] Reconnecting in {self._reconnect_delay:.1f}s...")
            time.sleep(self._reconnect_delay)

            # Exponential backoff
            self._reconnect_delay = min(
                self._reconnect_delay * 2,
                self._max_reconnect_delay
            )

            if self._running:
                self._start_websocket()

    def _submit_order(self, side: Side, order_type: str, quantity: Quantity,
                      price: Optional[Price], client_order_id: Optional[str]) -> OrderResponse:
        """Keep the former private execution entry point fail-closed."""
        raise NotImplementedError("Binance execution is disabled; use the local paper-trading account")


    def _signed_request(self, method: str, endpoint: str, params: Dict) -> Dict:
        """Prevent authenticated Binance REST requests from this project."""
        raise NotImplementedError("Authenticated Binance REST requests are disabled")


    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting using sliding window."""
        now = time.time()
        window = 60.0  # 1 minute window

        # Remove old requests outside window
        while not self._request_times.empty():
            try:
                oldest = self._request_times.queue[0]
                if now - oldest > window:
                    self._request_times.get_nowait()
                else:
                    break
            except Empty:
                break

        # Check if we're at limit
        if self._request_times.qsize() >= self.rate_limit:
            # Calculate sleep time
            oldest = self._request_times.queue[0]
            sleep_time = window - (now - oldest) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Record this request
        self._request_times.put(time.time())

    def _format_price(self, price: Optional[Price]) -> str:
        """Format price to Binance precision requirements."""
        if price is None:
            return ""
        # BTC/USDT typically uses 2 decimal places
        return f"{float(price):.2f}"

    def _format_quantity(self, quantity: Quantity) -> str:
        """Format quantity to Binance precision requirements."""
        # BTC/USDT typically uses 6 decimal places for quantity
        return f"{float(quantity):.6f}"
