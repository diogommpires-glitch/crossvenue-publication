"""
Simple Binance WebSocket Connector for Real-Time Market Data

Uses websocket-client for order book and trades streams.
Minimal dependencies for maximum portability.
"""

import json
import inspect
import threading
import time
from typing import Callable, Optional, Dict, List
import websocket


class BinanceWebSocket:
    """
    Real-time Binance market data via WebSocket.

    Streams:
    - Order book depth (best bid/ask + levels)
    - Recent trades (for paper trading fills)
    """

    def __init__(
        self,
        symbol: str = "btcusdt",
        on_orderbook: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
    ):
        self.symbol = symbol.lower()
        self.on_orderbook = on_orderbook
        self.on_trade = on_trade
        self._trade_callback_accepts_timestamp = self._callback_accepts_timestamp(on_trade)
        self._depth_callback_accepts_metadata = self._callback_accepts_arguments(on_orderbook, 3)
        self._trade_callback_accepts_metadata = self._callback_accepts_arguments(on_trade, 5)

        # WebSocket connections
        self.ws_depth = None
        self.ws_trades = None

        # Threads
        self.depth_thread = None
        self.trades_thread = None

        # State
        self.running = False
        self.connected = False
        self.depth_connected = False
        self.trades_connected = False
        self.last_depth_update_time = 0.0
        self.last_trade_update_time = 0.0

        # Stats
        self.orderbook_updates = 0
        self.trade_updates = 0
        self.last_update_time = 0

    @staticmethod
    def _callback_accepts_timestamp(callback: Optional[Callable]) -> bool:
        """Detect whether a legacy trade callback can receive the optional timestamp."""
        if callback is None:
            return False
        try:
            parameters = inspect.signature(callback).parameters.values()
        except (TypeError, ValueError):
            # Preserve the original three-argument callback contract when a callable's
            # signature cannot be inspected (for example, some extension callables).
            return False
        positional = [
            parameter for parameter in parameters
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        accepts_varargs = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        return accepts_varargs or len(positional) >= 4

    @staticmethod
    def _callback_accepts_arguments(callback: Optional[Callable], count: int) -> bool:
        """Detect opt-in metadata callback signatures without breaking legacy callers."""
        if callback is None:
            return False
        try:
            parameters = inspect.signature(callback).parameters.values()
        except (TypeError, ValueError):
            return False
        positional = [parameter for parameter in parameters if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )]
        # Variadic callbacks are ambiguous: many legacy callers use ``*args``
        # only to capture the historical payload. Require an explicit fixed
        # signature before adding metadata arguments.
        return len(positional) >= count

    def start(self):
        """Start WebSocket connections."""
        print(f"[WS] Connecting to Binance market data for {self.symbol.upper()}...")
        self.running = True

        # Start depth stream
        self.depth_thread = threading.Thread(target=self._run_depth_stream, daemon=True)
        self.depth_thread.start()

        # Start trades stream
        self.trades_thread = threading.Thread(target=self._run_trades_stream, daemon=True)
        self.trades_thread.start()

        # Wait for connection
        time.sleep(2)
        if self.connected:
            print(f"[WS] Connected to Binance streams")
        else:
            print(f"[WS] Warning: Connection may have failed")

    def stop(self):
        """Stop all WebSocket connections."""
        print(f"[WS] Disconnecting...")
        self.running = False

        if self.ws_depth:
            self.ws_depth.close()
        if self.ws_trades:
            self.ws_trades.close()
        current = threading.current_thread()
        for worker in (self.depth_thread, self.trades_thread):
            if worker and worker is not current:
                worker.join(timeout=2.0)
        self.depth_connected = False
        self.trades_connected = False
        self.connected = False
        print(f"[WS] Disconnected")

    def _run_depth_stream(self):
        """Run order book depth stream with controlled reconnection."""
        url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth20@100ms"

        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._handle_depth_update(data)
            except Exception as e:
                print(f"[WS] Depth error: {e}")

        def on_error(ws, error):
            print(f"[WS] Depth stream error: {error}")

        def on_close(ws, close_status_code, close_msg):
            self.depth_connected = False
            self.connected = self.depth_connected and self.trades_connected
            print(f"[WS] Depth stream closed")
            # Note: Reconnection handled by controlled loop in start(), not recursively here

        def on_open(ws):
            self.depth_connected = True
            self.connected = self.depth_connected and self.trades_connected
            print(f"[WS] Depth stream connected")

        # Controlled reconnection loop (replaces recursive reconnect)
        while self.running:
            self.ws_depth = websocket.WebSocketApp(
                url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
            )

            self.ws_depth.run_forever()

            # Only reconnect if still running
            if self.running:
                print(f"[WS] Reconnecting depth stream in 5s...")
                time.sleep(5)

    def _run_trades_stream(self):
        """Run recent trades stream with controlled reconnection."""
        url = f"wss://stream.binance.com:9443/ws/{self.symbol}@trade"

        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._handle_trade_update(data)
            except Exception as e:
                print(f"[WS] Trade error: {e}")

        def on_error(ws, error):
            print(f"[WS] Trade stream error: {error}")

        def on_close(ws, close_status_code, close_msg):
            self.trades_connected = False
            self.connected = self.depth_connected and self.trades_connected
            print(f"[WS] Trade stream closed")
            # Note: Reconnection handled by controlled loop, not recursively here

        def on_open(ws):
            self.trades_connected = True
            self.connected = self.depth_connected and self.trades_connected
            print(f"[WS] Trade stream connected")

        # Controlled reconnection loop (replaces recursive reconnect)
        while self.running:
            self.ws_trades = websocket.WebSocketApp(
                url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
            )

            self.ws_trades.run_forever()

            # Only reconnect if still running
            if self.running:
                print(f"[WS] Reconnecting trade stream in 5s...")
                time.sleep(5)

    def _handle_depth_update(self, data: Dict):
        """Process order book depth update."""
        self.orderbook_updates += 1
        self.last_update_time = time.time()
        self.last_depth_update_time = self.last_update_time

        if self.on_orderbook:
            try:
                # Extract bids and asks
                bids = [(float(price), float(qty)) for price, qty in data.get('bids', [])]
                asks = [(float(price), float(qty)) for price, qty in data.get('asks', [])]

                # Callback with parsed data
                if self._depth_callback_accepts_metadata:
                    exchange_time_ms = data.get('E')
                    metadata = {
                        "event_timestamp": (float(exchange_time_ms) / 1000.0
                                            if exchange_time_ms is not None else None),
                        "sequence": data.get('u', data.get('lastUpdateId')),
                    }
                    self.on_orderbook(bids, asks, metadata)
                else:
                    self.on_orderbook(bids, asks)
            except Exception as e:
                print(f"[WS] Error processing orderbook: {e}")

    def _handle_trade_update(self, data: Dict):
        """Process trade update."""
        self.trade_updates += 1
        self.last_trade_update_time = time.time()

        if self.on_trade:
            try:
                price = float(data['p'])
                quantity = float(data['q'])
                is_buyer_maker = data['m']  # True if buyer is maker (sell side)
                side = "sell" if is_buyer_maker else "buy"
                exchange_time_ms = data.get('T', data.get('E'))
                event_timestamp = float(exchange_time_ms) / 1000.0 if exchange_time_ms is not None else None
                if self._trade_callback_accepts_metadata:
                    self.on_trade(price, quantity, side, event_timestamp, data.get('t'))
                elif self._trade_callback_accepts_timestamp:
                    self.on_trade(price, quantity, side, event_timestamp)
                else:
                    self.on_trade(price, quantity, side)
            except Exception as e:
                print(f"[WS] Error processing trade: {e}")

    def get_stats(self) -> Dict:
        """Get connection statistics."""
        return {
            "connected": self.connected,
            "depth_connected": self.depth_connected,
            "trades_connected": self.trades_connected,
            "orderbook_updates": self.orderbook_updates,
            "trade_updates": self.trade_updates,
            "seconds_since_update": time.time() - self.last_update_time if self.last_update_time > 0 else None,
            "seconds_since_depth_update": time.time() - self.last_depth_update_time if self.last_depth_update_time > 0 else None,
            "seconds_since_trade_update": time.time() - self.last_trade_update_time if self.last_trade_update_time > 0 else None,
        }


# Legacy execution stub. No exchange execution is supported.
class BinanceREST:
    """
    Disabled compatibility stub for former order-execution experiments.
    """

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = ""
        self.api_secret = ""
        self.base_url = "https://api.binance.com"

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> Dict:
        """Reject live orders; the project uses local simulated execution."""
        raise NotImplementedError(
            "Binance order submission is disabled; use the local paper-trading account"
        )
