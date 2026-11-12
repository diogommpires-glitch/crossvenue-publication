#!/usr/bin/env python3
"""
BINANCE PUBLIC DATA PAPER-TRADING LAUNCHER WITH WEB UI

Orchestrates all components for paper trading with real-time market data:
- BinanceWebSocket: Real-time market data
- LiveTradingBridge: Strategy execution
- WebSocketUIServer: Real-time UI updates
- Browser interface: Trading terminal

Usage:
    python launch.py --capital 10000
    python launch.py --capital 50000 --no-browser
    python launch.py --capital 10000 --port 8765
"""

import sys
import os
import asyncio
import argparse
import time
import signal
import webbrowser
import json
import secrets
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.live_trading_bridge import LiveTradingBridge
from src.runtime.websocket_ui_server import WebSocketUIServer
from src.runtime.risk_manager import RiskManager, RiskLimits
from src.strategies.avellaneda_stoikov import AvellanedaStoikov
from src.connectors.binance_websocket import BinanceWebSocket



class AsyncBinanceWebSocket:
    """
    Async wrapper for BinanceWebSocket to integrate with async event loop.
    Bridges thread-based WebSocket to async UI updates.
    """

    def __init__(self, symbol: str, ui_server: WebSocketUIServer, bridge: LiveTradingBridge):
        self.symbol = symbol
        self.ui_server = ui_server
        self.bridge = bridge
        self.ws = None
        self.running = False
        self.loop = None  # Store event loop reference

    def start(self):
        """Start WebSocket with callbacks that trigger UI updates."""
        # Capture the running event loop
        self.loop = asyncio.get_running_loop()

        async def on_orderbook_wrapper(bids, asks):
            """Wrapper to handle orderbook and update UI."""
            # Update bridge
            try:
                self.bridge.on_orderbook_update(bids, asks)
            except Exception as exc:
                print(f"[PAPER] Depth update failed in the paper bridge: {exc}")

            # Store market data for UI
            if bids and asks:
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                mid_price = (best_bid + best_ask) / 2.0
                spread = best_ask - best_bid

                self.ui_server.last_market_data = {
                    "mid_price": mid_price,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "spread_bps": (spread / mid_price) * 10000,
                }

                # The built-in terminal expects an orderbook payload, not just
                # a mid-price summary. Convert Binance tuples to its row shape.
                ui_bids = [
                    {"price": price, "size": quantity, "total": price * quantity}
                    for price, quantity in bids
                ]
                ui_asks = [
                    {"price": price, "size": quantity, "total": price * quantity}
                    for price, quantity in asks
                ]
                self.ui_server.last_market_data["bids"] = ui_bids
                self.ui_server.last_market_data["asks"] = ui_asks
                await self.ui_server.broadcast_update("orderbook", {
                    "bids": ui_bids,
                    "asks": ui_asks,
                    **self.ui_server.last_market_data,
                })
                await self.ui_server.broadcast_update("price", {
                    "price": mid_price,
                })

        async def on_trade_wrapper(price, quantity, side, event_timestamp=None):
            """Wrapper to handle trades and update UI."""
            # Update bridge
            try:
                self.bridge.on_trade(price, quantity, side, event_timestamp)
            except Exception as exc:
                print(f"[PAPER] Trade update failed in the paper bridge: {exc}")

            # Broadcast trade to UI
            await self.ui_server.broadcast_update("trade", {
                "price": price,
                "quantity": quantity,
                "size": quantity,
                "side": side,
                "timestamp": datetime.now().isoformat(),
                "event_timestamp": event_timestamp,
            })

        def sync_orderbook_callback(bids, asks):
            """Sync callback that schedules async UI update from thread."""
            if self.loop and not self.loop.is_closed():
                self._schedule_ui_update(on_orderbook_wrapper(bids, asks), "depth")

        def sync_trade_callback(price, quantity, side, event_timestamp=None):
            """Sync callback that schedules async UI update from thread."""
            if self.loop and not self.loop.is_closed():
                self._schedule_ui_update(
                    on_trade_wrapper(price, quantity, side, event_timestamp), "trade"
                )

        # Create WebSocket with wrapped callbacks
        self.ws = BinanceWebSocket(
            symbol=self.symbol,
            on_orderbook=sync_orderbook_callback,
            on_trade=sync_trade_callback,
        )

        # Store reference in UI server
        self.ui_server.ws_market = self.ws

        # Start WebSocket
        self.ws.start()
        self.running = True

    def _schedule_ui_update(self, coroutine, stream_name: str) -> None:
        """Report thread-to-async callback failures instead of silently losing feed updates."""
        if self.loop is None or self.loop.is_closed():
            coroutine.close()
            return
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        except RuntimeError as exc:
            coroutine.close()
            print(f"[UI] Failed to schedule Binance {stream_name} update: {exc}")
            return

        def report_failure(completed):
            try:
                completed.result()
            except Exception as exc:
                print(f"[UI] Failed to forward Binance {stream_name} update: {exc}")

        future.add_done_callback(report_failure)

    def stop(self):
        """Stop WebSocket."""
        if self.ws:
            self.ws.stop()
        self.running = False


def create_trading_terminal_html(port: int, output_path: Optional[Path] = None) -> str:
    """Generate the canonical Phase 3 paper dashboard with the active local port."""
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    template_path = PROJECT_ROOT / "assets/terminal.html"
    html_path = Path(output_path) if output_path is not None else PROJECT_ROOT / "trading_terminal.html"
    template = template_path.read_text(encoding="utf-8")
    assets = PROJECT_ROOT / "assets"
    stylesheet_paths = [assets / name for name in ("tokens.css", "base.css", "terminal.css")]
    if any(not path.is_file() for path in stylesheet_paths):
        raise ValueError("Dashboard stylesheet is missing; restore assets/tokens.css, base.css, and terminal.css")
    tokens, base, stylesheet = [path.read_text(encoding="utf-8") for path in stylesheet_paths]
    stamp, rules = stylesheet.split('@import url("./tokens.css");\n@import url("./base.css");', 1)
    template = template.replace('<link rel="stylesheet" href="terminal.css">',
        "<style>" + stamp + tokens + base + rules + "</style>", 1)
    chart_asset = PROJECT_ROOT / "assets" / "chart.umd.min.js"
    if not chart_asset.is_file():
        raise ValueError("Bundled chart asset is missing; restore assets/chart.umd.min.js")
    chart_script = "<script>" + chart_asset.read_text(encoding="utf-8") + "</script>"
    for chart_reference in (
            '<script src="assets/chart.umd.min.js"></script>',
            '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>'):
        template = template.replace(chart_reference, chart_script, 1)
    marker = "const PAPER_WS_PORT = 8765;"
    if marker not in template:
        raise ValueError("Phase 3 dashboard is missing its WebSocket port marker")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(template.replace(marker, f"const PAPER_WS_PORT = {port};", 1), encoding="utf-8")
    return str(html_path)


async def run_trading_system(capital: float, port: int, open_browser: bool):
    """
    Main orchestration function for entire trading system.

    Coordinates:
    - WebSocket UI Server
    - Binance market data
    - Trading bridge and strategy
    - Browser interface
    """

    print("=" * 80)
    print(" " * 20 + "PAPER TRADING TERMINAL WITH WEB UI")
    print("=" * 80)
    print()
    print(f"Paper Trading Mode - Safe simulation with real market data")
    print(f"Starting Capital: ${capital:,.2f}")
    print()

    # Setup components
    print("[INIT] Initializing trading system...")
    print()

    # Risk limits
    limits = RiskLimits(
        max_position_size=0.1,
        max_order_size=0.01,
        max_daily_loss=capital * 0.1,  # 10% max loss
        max_total_exposure=capital,
        max_orders_per_minute=60,  # Increased from 10 to allow frequent updates
        circuit_breaker_loss_pct=0.05,
        min_balance_reserve=capital * 0.05,  # 5% reserve
    )

    risk_manager = RiskManager(limits)

    # Strategy
    strategy = AvellanedaStoikov(
        risk_aversion=0.05,
        volatility=0.20,
        order_size=0.01,
        max_inventory=0.1,
        time_horizon=3600.0,
        strategy_id="mm_strategy",
    )

    # Trading bridge
    bridge = LiveTradingBridge(
        strategy=strategy,
        risk_manager=risk_manager,
        connector=None,
        paper_mode=True,
        starting_cash=capital,
    )
    bridge.strategy_interval = 5.0  # Update every 5 seconds instead of 1 second

    # Create UI Server FIRST (so callbacks can reference it)
    session_token = secrets.token_urlsafe(32)
    ui_server = WebSocketUIServer(bridge=bridge, port=port, session_token=session_token)

    # Capture event loop NOW (we're in async context)
    event_loop = asyncio.get_running_loop()

    # Wire bridge events to UI updates
    def forward_fill_to_ui(fill_data):
        """Forward strategy fills to UI."""
        try:
            asyncio.run_coroutine_threadsafe(
                ui_server.broadcast_update("strategy_fill", fill_data),
                event_loop
            )
        except Exception as e:
            print(f"[UI] Error forwarding fill: {e}")

    def forward_pnl_to_ui(pnl_data):
        """Forward P&L updates to UI."""
        try:
            with bridge.orders_lock:
                total_completed_orders = len(bridge.filled_order_ids)
            position_data = {
                "position": pnl_data.get("position", bridge.position),
                "cash": pnl_data.get("cash", bridge.cash),
                "equity": pnl_data.get("equity", bridge.starting_cash),
                "total_pnl": pnl_data.get("total_pnl", 0),
                "total_fees": pnl_data.get("total_fees", 0),
                "realized_pnl": pnl_data.get("realized_pnl", 0),
                "unrealized_pnl": pnl_data.get("unrealized_pnl", 0),
                "total_trades": len(bridge.trades),
                "total_orders": bridge.total_orders_sent,
                "total_fills": bridge.total_fills,
                "total_completed_orders": total_completed_orders,
                "total_rejections": bridge.total_rejections,
                "total_traded_notional": bridge.total_traded_notional,
            }
            asyncio.run_coroutine_threadsafe(
                ui_server.broadcast_update("position", position_data),
                event_loop
            )
        except Exception as e:
            print(f"[UI] Error forwarding P&L: {e}")

    def forward_performance_to_ui(perf_data):
        """Forward performance metrics to UI."""
        try:
            asyncio.run_coroutine_threadsafe(
                ui_server.broadcast_update("performance_metrics", perf_data),
                event_loop
            )
        except Exception as e:
            print(f"[UI] Error forwarding performance metrics: {e}")

    def forward_active_orders_to_ui(orders_data):
        """Forward active orders to UI."""
        try:
            asyncio.run_coroutine_threadsafe(
                ui_server.broadcast_update("active_orders", orders_data),
                event_loop
            )
        except Exception as e:
            print(f"[UI] Error forwarding active orders: {e}")

    def forward_cancelled_orders_to_ui(cancellation_data):
        """Forward paper-order cancellations so the dashboard clears stale quotes."""
        try:
            asyncio.run_coroutine_threadsafe(
                ui_server.broadcast_update("orders_cancelled", cancellation_data),
                event_loop
            )
        except Exception as e:
            print(f"[UI] Error forwarding cancelled orders: {e}")

    # Register callbacks
    bridge.on("fill", forward_fill_to_ui)
    bridge.on("pnl_update", forward_pnl_to_ui)
    bridge.on("performance_metrics", forward_performance_to_ui)
    bridge.on("active_orders", forward_active_orders_to_ui)
    bridge.on("orders_cancelled", forward_cancelled_orders_to_ui)

    async_ws = None
    try:
        # Start the UI before market data so the browser can connect during feed startup.
        await ui_server.start_server()
        port = ui_server.port
        html_path = create_trading_terminal_html(port)
        html_file = Path(html_path)
        document = html_file.read_text(encoding="utf-8")
        token_marker = "const PAPER_SESSION_TOKEN = '';"
        if token_marker not in document:
            raise RuntimeError("Dashboard session marker is missing; restore the matching UI template")
        html_file.write_text(document.replace(token_marker,
            f"const PAPER_SESSION_TOKEN = {json.dumps(session_token)};", 1), encoding="utf-8")
        print(f"[UI] Trading terminal created: {html_path}")
        async_ws = AsyncBinanceWebSocket(symbol="btcusdt", ui_server=ui_server, bridge=bridge)
        print("[BINANCE] Connecting to BTC/USDT public streams...")
        async_ws.start()
    except BaseException:
        if async_ws is not None:
            async_ws.stop()
        bridge.stop_strategy()
        await ui_server.stop_server()
        raise

    # Wait for a valid public book before generating any quotes.
    try:
        deadline = time.monotonic() + 15.0
        while not bridge.market_data_is_fresh() and time.monotonic() < deadline:
            await asyncio.sleep(0.25)
    except BaseException:
        bridge.stop_strategy()
        if async_ws is not None:
            async_ws.stop()
        await ui_server.stop_server()
        raise
    if ui_server.start_when_ready():
        print("[STRATEGY] Fresh Binance depth and trade data received; starting paper strategy")
    else:
        print("[STRATEGY] Waiting for fresh Binance market data; no paper orders will be placed")

    print()
    print("=" * 80)
    print("[STRATEGY] Paper-only mode")

    # Open browser
    try:
        if open_browser:
            print("[BROWSER] Opening trading terminal...")
            webbrowser.open(f'file://{html_path}')
        else:
            print(f"[BROWSER] Skipped (use --no-browser). Open manually: file://{html_path}")
    except BaseException:
        bridge.stop_strategy()
        if async_ws is not None:
            async_ws.stop()
        await ui_server.stop_server()
        raise

    # Cancellation during feed wait/browser startup must close both data and UI workers.

    print()
    print("=" * 80)
    print("SYSTEM RUNNING")
    print("=" * 80)
    print()
    print(f"Web UI: ws://127.0.0.1:{port}")
    print(f"Trading Terminal: file://{html_path}")
    print(f"Press Ctrl+C to stop")
    print()
    print("=" * 80)
    print()

    # Status monitoring loop
    last_status_time = time.time()
    status_interval = 60.0
    last_feed_readiness = None
    last_stream_connectivity = None

    try:
        while True:
            await asyncio.sleep(1)
            ui_server.start_when_ready()
            feed_status = bridge.get_status()
            feed_status["websocket"] = async_ws.ws.get_stats() if async_ws and async_ws.ws else {}
            readiness = (feed_status["book_ready"], feed_status["trade_ready"])
            websocket_stats = feed_status["websocket"]
            stream_connectivity = (
                websocket_stats.get("depth_connected", False),
                websocket_stats.get("trades_connected", False),
            )
            if readiness != last_feed_readiness or stream_connectivity != last_stream_connectivity:
                await ui_server.broadcast_update("status", feed_status)
                last_feed_readiness = readiness
                last_stream_connectivity = stream_connectivity

            # Periodic console status
            now = time.time()
            if now - last_status_time >= status_interval:
                status = bridge.get_status()

                print(f"\n[STATUS] {time.strftime('%H:%M:%S')}")
                print(f"  Uptime: {status['uptime_seconds']/60:.1f}m")
                print(f"  Position: {status['position']:.4f} BTC")
                print(f"  Cash: ${status['cash']:.2f}")
                print(f"  Net P&L: ${status['total_pnl']:.2f}")
                print(f"  Completed orders: {status['total_completed_orders']} | Orders accepted: {status['total_orders_sent']} | Fill events: {status['total_fills']}")
                print(f"  UI Clients: {len(ui_server.clients)}")
                print()

                # Broadcast status update to UI
                await ui_server.broadcast_update("status", {
                    "uptime": status['uptime_seconds'],
                    "clients": len(ui_server.clients),
                    "book_ready": status["book_ready"],
                    "trade_ready": status["trade_ready"],
                    "market_data_ready": status["market_data_ready"],
                    "is_running": status["is_running"],
                    "websocket": feed_status["websocket"],
                })

                last_status_time = now

    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Graceful shutdown initiated...")
    finally:
        bridge.stop_strategy()
        if async_ws is not None:
            async_ws.stop()
        await ui_server.stop_server()

    status = bridge.get_status()
    print()
    print("=" * 80)
    print("FINAL PAPER ACCOUNT STATUS")
    print("=" * 80)
    print(f"Uptime: {status['uptime_seconds']/60:.1f} minutes")
    print(f"Position: {status['position']:.4f} BTC")
    print(f"Cash: ${status['cash']:.2f}")
    print(f"Net P&L: ${status['total_pnl']:.2f}")
    print(f"Total fees: ${status['total_fees']:.2f}")
    print(f"Completed orders: {status['total_completed_orders']}")
    print(f"Orders accepted: {status['total_orders_sent']}")
    print(f"Fill events: {status['total_fills']}")
    print(f"Paper traded notional: ${status['total_traded_notional']:.2f}")
    print(f"Rejections: {status['total_rejections']}")
    print("=" * 80)


def main():
    """Parse arguments and launch trading system."""
    parser = argparse.ArgumentParser(
        description="Paper Trading Terminal with live public Binance data"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Starting capital in USDT (default: 10000)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Run configured multi-asset local paper mode from a JSON config",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.capital <= 0:
        print("[ERROR] Capital must be positive")
        sys.exit(1)

    if args.port < 1024 or args.port > 65535:
        print("[ERROR] Port must be between 1024 and 65535")
        sys.exit(1)

    # Run async main
    try:
        if args.config is not None:
            from src.runtime.cross_asset_app import run_cross_asset_system
            asyncio.run(run_cross_asset_system(
                config_path=str(args.config),
                port=args.port,
                open_browser=not args.no_browser,
                html_factory=create_trading_terminal_html,
                project_root=PROJECT_ROOT,
            ))
        else:
            asyncio.run(run_trading_system(
                capital=args.capital,
                port=args.port,
                open_browser=not args.no_browser,
            ))
    except KeyboardInterrupt:
        print("\n[EXIT] Stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
