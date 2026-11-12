"""Single loopback WebSocket backend for the paper-trading terminal."""

import asyncio
import json
import hmac
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urlsplit

import websockets


class WebSocketUIServer:
    """Serve the launcher terminal protocol on loopback and forward bridge events."""

    def __init__(self, bridge=None, host: str = "127.0.0.1", port: int = 8765,
                 broadcast_interval: float = 0.1, application_service=None,
                 session_token: Optional[str] = None, max_message_bytes: int = 65536):
        if host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("The dashboard server must bind to loopback")
        if not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer between 0 and 65535")
        if broadcast_interval <= 0:
            raise ValueError("broadcast_interval must be positive")
        self.bridge = bridge
        self.host = host
        self.port = port
        self.broadcast_interval = float(broadcast_interval)
        self.clients: Set = set()
        self.clients_lock = asyncio.Lock()
        self.server = None
        self.running = False
        self.ws_market = None
        self.last_market_data: Dict = {}
        self.recent_market_trades: List[Dict] = []
        self.auto_start_pending = True
        self.application_service = application_service
        self.session_token = session_token
        self.max_message_bytes = int(max_message_bytes)
        if self.max_message_bytes < 1024:
            raise ValueError("max_message_bytes must be at least 1024")

    async def start(self):
        if self.running:
            return
        self.server = await websockets.serve(self.handler, self.host, self.port,
                                             ping_interval=20, ping_timeout=10,
                                             max_size=self.max_message_bytes, max_queue=16)
        self.running = True
        if self.port == 0 and self.server.sockets:
            self.port = int(self.server.sockets[0].getsockname()[1])

    async def start_server(self):
        await self.start()

    async def stop(self):
        self.running = False
        async with self.clients_lock:
            clients = list(self.clients)
        if clients:
            await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def stop_server(self):
        await self.stop()

    async def register_client(self, websocket):
        async with self.clients_lock:
            self.clients.add(websocket)
        await self.send_full_state(websocket)

    async def unregister_client(self, websocket):
        async with self.clients_lock:
            self.clients.discard(websocket)

    async def send_full_state(self, websocket):
        status = self.bridge.get_full_status() if self.bridge is not None else {}
        multi_asset_state = None
        if self.bridge is not None and callable(getattr(self.bridge, "get_dashboard_state", None)):
            multi_asset_state = self.bridge.get_dashboard_state()
            status = self.bridge.get_status()
        ws_stats = self.ws_market.get_stats() if self.ws_market is not None else {}
        trading = {
            "position": status.get("position", 0.0), "cash": status.get("cash", 0.0),
            "realized_pnl": status.get("realized_pnl", 0.0),
            "unrealized_pnl": status.get("unrealized_pnl", 0.0),
            "total_pnl": status.get("total_pnl", 0.0), "total_fees": status.get("total_fees", 0.0),
            "equity": status.get("equity", 0.0), "total_trades": status.get("total_trades", 0),
            "total_orders": status.get("total_orders_sent", 0), "total_fills": status.get("total_fills", 0),
            "total_completed_orders": status.get("total_completed_orders", 0),
            "total_traded_notional": status.get("total_traded_notional", 0.0),
            "total_rejections": status.get("total_rejections", 0),
            "active_orders": status.get("active_order_count", status.get("active_orders", 0)),
            "uptime_seconds": status.get("uptime_seconds", 0.0),
        }
        market = dict(self.last_market_data)
        if multi_asset_state is not None:
            instruments = multi_asset_state.get("instruments", [])
            selected = next((item for item in instruments
                             if item.get("asset_class") == "crypto"), None)
            selected = selected or (instruments[0] if instruments else {})
            if selected:
                market = {
                    "instrument_id": selected.get("instrument_id"),
                    "mid_price": selected.get("mid_price"),
                    "best_bid": selected.get("best_bid"),
                    "best_ask": selected.get("best_ask"),
                }
                status["position"] = selected.get("position", 0.0)
                status["active_orders"] = selected.get("active_orders", [])
            status["active_order_count"] = multi_asset_state.get("totals", {}).get("active_orders", 0)
            trading.update({
                "position": status.get("position", 0.0),
                "cash": (multi_asset_state.get("portfolio") or {}).get("cash_by_currency", {}).get(
                    status.get("reporting_currency", "USD"), 0.0),
                "equity": (multi_asset_state.get("portfolio") or {}).get("equity", 0.0),
                "total_pnl": (multi_asset_state.get("portfolio") or {}).get("total_pnl", 0.0),
                "realized_pnl": (multi_asset_state.get("portfolio") or {}).get("realized_pnl", 0.0),
                "unrealized_pnl": (multi_asset_state.get("portfolio") or {}).get("unrealized_pnl", 0.0),
                "total_fees": (multi_asset_state.get("portfolio") or {}).get("fees_paid", 0.0),
                "total_fills": status.get("total_fills", 0),
                "total_orders": status.get("total_orders_sent", 0),
                "total_rejections": status.get("total_rejections", 0),
                "total_traded_notional": status.get("total_traded_notional", 0.0),
                "active_orders": status.get("active_orders_detail", []),
            })
        if not market and status.get("mid_price"):
            market = {"mid_price": status["mid_price"]}
        await websocket.send(json.dumps({
            "type": "full_state", "timestamp": datetime.now().isoformat(),
            "data": status, "trading": trading, "market": market,
            "websocket": ws_stats, "market_trades": list(self.recent_market_trades),
            "multi_asset_state": multi_asset_state,
        }))

    async def handler(self, websocket, path=None):
        if self.session_token is not None:
            headers = getattr(websocket, "request_headers", None)
            if headers is None:
                headers = getattr(getattr(websocket, "request", None), "headers", {})
            if not self._origin_allowed(headers.get("Origin")):
                await websocket.close(code=1008, reason="Local dashboard origin required")
                return
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=10)
                if len(raw.encode("utf-8")) > self.max_message_bytes:
                    raise ValueError("Authentication payload exceeds limit")
                authentication = json.loads(raw)
                supplied = authentication.get("token", "")
                if (authentication.get("type") != "authenticate" or not isinstance(supplied, str)
                        or not hmac.compare_digest(supplied, self.session_token)):
                    raise ValueError("Authentication failed")
            except Exception:
                await websocket.close(code=1008, reason="Dashboard session authentication required")
                return
            await websocket.send(json.dumps({"type": "authenticated", "protocol_version": 1}))
        await self.register_client(websocket)
        try:
            async for raw in websocket:
                await self._handle_message(websocket, raw)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

    async def _handle_message(self, websocket, raw: str):
        try:
            if not isinstance(raw, str) or len(raw.encode("utf-8")) > self.max_message_bytes:
                raise ValueError("Paper command exceeds the message limit")
            message = json.loads(raw)
            if not isinstance(message, dict):
                raise ValueError("Paper command must be an object")
            kind = message.get("type")
            payload = message.get("data", {})
            if kind == "ping":
                await websocket.send(json.dumps({"type": "pong"}))
                return
            if self.application_service is not None:
                if kind != "paper_command":
                    raise ValueError("This application requires the versioned paper command protocol")
                result = await asyncio.wrap_future(self.application_service.submit_command(payload))
                await websocket.send(json.dumps({"type": "control_response", **result}))
                await self.broadcast_update("multi_asset_state", self.application_service.get_dashboard_state())
                return
            actions = {
                "start_strategy": "start", "start": "start",
                "stop_strategy": "stop", "stop": "stop",
                "reset_strategy": "reset", "reset": "reset",
            }
            action = actions.get(kind)
            if action == "start":
                self.auto_start_pending = False
                self.bridge.start_strategy()
                result = {"message": "Paper strategy started"}
            elif action == "stop":
                self.auto_start_pending = False
                self.bridge.stop_strategy()
                result = {"message": "Paper strategy stopped"}
            elif action == "reset":
                self.auto_start_pending = False
                self.bridge.reset_strategy()
                result = {"message": "Paper account reset"}
            elif kind == "acknowledge_risk_circuit_breaker":
                acknowledge = getattr(self.bridge, "acknowledge_risk_circuit_breaker", None)
                if not callable(acknowledge):
                    raise ValueError("risk-halt acknowledgment is unavailable in this paper mode")
                self.auto_start_pending = False
                result = acknowledge()
            elif kind == "update_params":
                params = message.get("params", payload.get("params", payload))
                self.bridge.update_parameters(params)
                result = {"updated": params}
            elif kind == "roll_future":
                roll = getattr(self.bridge, "roll_future", None)
                if not callable(roll):
                    raise ValueError("futures rolling is unavailable in this paper mode")
                result = roll(payload.get("old_instrument_id"), payload.get("new_instrument_id"))
            elif kind == "apply_equity_split":
                apply_split = getattr(self.bridge, "apply_equity_split", None)
                if not callable(apply_split):
                    raise ValueError("equity corporate actions are unavailable in this paper mode")
                result = apply_split(
                    payload.get("action_id"), payload.get("instrument_id"),
                    payload.get("ratio"), timestamp=payload.get("timestamp"))
            elif kind == "apply_equity_dividend":
                apply_dividend = getattr(self.bridge, "apply_equity_dividend", None)
                if not callable(apply_dividend):
                    raise ValueError("equity corporate actions are unavailable in this paper mode")
                result = apply_dividend(
                    payload.get("action_id"), payload.get("instrument_id"),
                    payload.get("amount_per_share"), payload.get("eligible_quantity"),
                    timestamp=payload.get("timestamp"))
            elif kind == "record_fx_financing":
                record_financing = getattr(self.bridge, "record_fx_financing", None)
                if not callable(record_financing):
                    raise ValueError("rolling-FX financing is unavailable in this paper mode")
                result = record_financing(
                    payload.get("adjustment_id"), payload.get("instrument_id"),
                    payload.get("amount"), payload.get("currency"),
                    timestamp=payload.get("timestamp"), kind=payload.get("kind", "fx_financing"))
            elif kind in {"get_status", "status_request"}:
                result = self.bridge.get_dashboard_state() if callable(
                    getattr(self.bridge, "get_dashboard_state", None)) else self.bridge.get_full_status()
            elif kind == "get_trades":
                limit = int(payload.get("limit", 50))
                get_trades = getattr(self.bridge, "get_trades", None)
                result = get_trades(limit) if callable(get_trades) else {
                    "trades": self.bridge.trades[-max(0, limit):], "total": len(self.bridge.trades)}
            else:
                raise ValueError(f"Unknown command type: {kind}")
            await websocket.send(json.dumps({"type": "control_response", "action": kind,
                                             "success": True, "data": result}))
            if action is not None or kind in {
                    "update_params", "acknowledge_risk_circuit_breaker",
                    "apply_equity_split", "apply_equity_dividend", "record_fx_financing"}:
                get_dashboard_state = getattr(self.bridge, "get_dashboard_state", None)
                if callable(get_dashboard_state):
                    await self.broadcast_update("multi_asset_state", get_dashboard_state())
                else:
                    await self.broadcast_update("status", self.bridge.get_full_status())
        except websockets.exceptions.ConnectionClosed:
            return
        except Exception as exc:
            await websocket.send(json.dumps({"type": "error", "message": str(exc), "success": False}))

    @staticmethod
    def _origin_allowed(origin):
        # File dashboards send a null origin; they still need the random
        # instance token before any state or commands are accessible.
        if origin in (None, "null"):
            return True
        try:
            parsed = urlsplit(origin)
            return (parsed.scheme in {"http", "https"} and parsed.hostname in {
                "localhost", "127.0.0.1", "::1"} and parsed.username is None
                and parsed.password is None and parsed.path in {"", "/"})
        except (ValueError, TypeError):
            return False

    def start_when_ready(self) -> bool:
        """Start once on fresh feeds unless the user has already taken control."""
        if not self.auto_start_pending or self.bridge is None:
            return False
        if getattr(self.bridge, "circuit_breaker_active", False):
            return False
        if not self.bridge.market_data_is_fresh():
            return False
        self.auto_start_pending = False
        self.bridge.start_strategy()
        return True

    async def broadcast_update(self, update_type: str, data: dict):
        if update_type == "trade":
            self.recent_market_trades.append(dict(data))
            self.recent_market_trades = self.recent_market_trades[-20:]
        await self._broadcast({"type": update_type, "timestamp": datetime.now().isoformat(), "data": data})

    async def broadcast_market_data(self, bids: List[tuple], asks: List[tuple]):
        mid = (float(bids[0][0]) + float(asks[0][0])) / 2 if bids and asks else 0.0
        spread = float(asks[0][0]) - float(bids[0][0]) if bids and asks else 0.0
        self.last_market_data = {"mid_price": mid, "best_bid": float(bids[0][0]) if bids else 0.0,
                                 "best_ask": float(asks[0][0]) if asks else 0.0, "spread": spread,
                                 "spread_bps": spread / mid * 10000 if mid else 0.0,
                                 "bids": [{"price": float(p), "size": float(q), "total": float(p) * float(q)} for p, q in bids],
                                 "asks": [{"price": float(p), "size": float(q), "total": float(p) * float(q)} for p, q in asks]}
        await self.broadcast_update("orderbook", {"bids": [{"price": float(p), "size": float(q), "total": float(p) * float(q)} for p, q in bids],
                                                   "asks": [{"price": float(p), "size": float(q), "total": float(p) * float(q)} for p, q in asks],
                                                   **self.last_market_data})

    async def broadcast_trade(self, price: float, quantity: float, side: str):
        await self.broadcast_update("trade", {"price": float(price), "quantity": float(quantity),
                                               "size": float(quantity), "side": side})

    async def broadcast_status(self, status_dict: Dict):
        await self.broadcast_update("status", status_dict)

    async def _broadcast(self, message: Dict):
        async with self.clients_lock:
            clients = list(self.clients)
        if not clients:
            return
        encoded = json.dumps(message)
        results = await asyncio.gather(*(client.send(encoded) for client in clients), return_exceptions=True)
        failed = {client for client, result in zip(clients, results) if isinstance(result, Exception)}
        if failed:
            async with self.clients_lock:
                self.clients.difference_update(failed)


async def run_server(bridge, host: str = "127.0.0.1", port: int = 8765):
    server = WebSocketUIServer(bridge=bridge, host=host, port=port)
    try:
        await server.start()
        await asyncio.Future()
    finally:
        await server.stop()
