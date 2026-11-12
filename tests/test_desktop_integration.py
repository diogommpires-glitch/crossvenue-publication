import time
import json
from types import SimpleNamespace

import pytest

from launch import create_trading_terminal_html
from src.orderbook.order import Order
from src.runtime.live_trading_bridge import LiveTradingBridge
from src.runtime.risk_manager import RiskLimits, RiskManager
from src.runtime.websocket_ui_server import WebSocketUIServer
from src.strategies.avellaneda_stoikov import AvellanedaStoikov
from src.utils.types import OrderType, Price, Quantity, Side, Timestamp


def make_bridge(capital=10_000.0, circuit_breaker=0.05, stale_after=10.0):
    strategy = AvellanedaStoikov(risk_aversion=0.05, volatility=0.2, order_size=0.01,
                                 max_inventory=1.0, strategy_id="paper-test")
    risk = RiskManager(RiskLimits(max_position_size=2.0, max_order_size=1.0,
                                  max_daily_loss=capital, max_total_exposure=capital * 10,
                                  max_orders_per_minute=100, circuit_breaker_loss_pct=circuit_breaker,
                                  min_balance_reserve=0.0))
    return LiveTradingBridge(strategy, risk, paper_mode=True, starting_cash=capital,
                             market_data_stale_after=stale_after)


def make_order(order_id, side=Side.BUY, qty=0.4, price=100.0):
    return Order(order_id=order_id, side=side, order_type=OrderType.LIMIT,
                 quantity=Quantity(qty), price=Price(price), timestamp=Timestamp(time.time()),
                 trader_id="paper-test")


def test_custom_capital_initializes_every_ledger_consistently():
    bridge = make_bridge(capital=2_500.0)
    bridge.on_orderbook_update([(99.0, 1.0)], [(101.0, 1.0)])
    status = bridge.get_status()
    assert status["equity"] == pytest.approx(2_500.0)
    assert status["total_pnl"] == pytest.approx(0.0)
    assert bridge.pnl_tracker.starting_capital == pytest.approx(2_500.0)
    assert bridge.risk_analytics.starting_equity == pytest.approx(2_500.0)
    assert bridge.risk_manager.get_status()["daily_pnl"] == pytest.approx(0.0)
    assert not bridge.risk_manager.circuit_breaker_active


def test_accounting_marks_inventory_and_charges_maker_fees(monkeypatch):
    bridge = make_bridge()
    bridge.on_orderbook_update([(109.0, 1.0)], [(111.0, 1.0)])
    bridge._process_strategy_order(make_order("buy-1", qty=0.1, price=100.0), 110.0)
    later = max(bridge.order_timestamps.values()) + 0.01
    monkeypatch.setattr(time, "time", lambda: later)
    bridge.is_running = True
    bridge.on_trade(price=100.0, quantity=0.2, side="sell", event_timestamp=time.time())
    status = bridge.get_status()
    assert status["position"] == pytest.approx(0.1)
    assert status["total_fees"] == pytest.approx(0.01)
    assert status["equity"] == pytest.approx(10_000.0 + status["total_pnl"])
    assert status["unrealized_pnl"] == pytest.approx(1.0)
    bridge.is_running = False


def test_one_public_trade_volume_is_shared_across_all_paper_orders(monkeypatch):
    bridge = make_bridge()
    bridge.on_orderbook_update([(99.0, 1.0)], [(101.0, 1.0)])
    bridge._process_strategy_order(make_order("buy-a", qty=0.4), 100.0)
    bridge._process_strategy_order(make_order("buy-b", qty=0.4), 100.0)
    later = max(bridge.order_timestamps.values()) + 0.01
    monkeypatch.setattr(time, "time", lambda: later)
    bridge.is_running = True
    bridge.on_trade(price=100.0, quantity=0.5, side="sell", event_timestamp=time.time())
    assert sum(float(trade["quantity"]) for trade in bridge.trades) <= 0.25 + 1e-12
    assert bridge.position == pytest.approx(0.25)
    assert len(bridge.active_orders) == 2
    assert bridge.get_status()["total_completed_orders"] == 0
    assert bridge.calculate_fill_rate() == 0.0
    bridge.on_trade(price=100.0, quantity=0.3, side="sell", event_timestamp=time.time())
    assert bridge.calculate_fill_rate() == pytest.approx(50.0)
    status = bridge.get_status()
    assert status["total_fills"] == 2
    assert status["total_completed_orders"] == 1
    assert status["total_traded_notional"] == pytest.approx(40.0)
    bridge.is_running = False
    bridge._cancel_all_quotes()


def test_circuit_breaker_cancels_quotes_before_next_trade_can_fill(monkeypatch):
    bridge = make_bridge(capital=100.0, circuit_breaker=0.01)
    bridge.on_orderbook_update([(89.0, 1.0)], [(91.0, 1.0)])
    bridge._process_strategy_order(make_order("buy-a", qty=0.1), 90.0)
    bridge._process_strategy_order(make_order("buy-b", qty=0.1), 90.0)
    later = max(bridge.order_timestamps.values()) + 0.01
    monkeypatch.setattr(time, "time", lambda: later)
    bridge.is_running = True
    bridge.on_trade(price=100.0, quantity=0.4, side="sell", event_timestamp=time.time())
    assert bridge.risk_manager.circuit_breaker_active
    assert bridge.total_fills == 1
    assert bridge.active_orders == {}


def test_stale_orderbook_cancels_quotes_and_prevents_paper_fill():
    bridge = make_bridge(stale_after=0.01)
    bridge.on_orderbook_update([(99.0, 1.0)], [(101.0, 1.0)])
    bridge._process_strategy_order(make_order("buy-stale"), 100.0)
    bridge.is_running = True
    bridge.last_orderbook_time = time.time() - 1.0
    bridge.on_trade(price=99.0, quantity=1.0, side="sell")
    assert bridge.total_fills == 0
    assert bridge.active_orders == {}


def test_live_execution_is_rejected_and_does_not_call_connector():
    strategy = AvellanedaStoikov(strategy_id="paper-test")
    risk = RiskManager(RiskLimits())
    with pytest.raises(ValueError, match="paper trading only"):
        LiveTradingBridge(strategy, risk, connector=SimpleNamespace(place_order=lambda *_: pytest.fail()),
                          paper_mode=False)


def test_dashboard_launcher_and_shared_backend_protocol_are_aligned():
    import launch

    assert launch.WebSocketUIServer is WebSocketUIServer
    server = WebSocketUIServer(bridge=make_bridge())
    assert server.host in {"127.0.0.1", "localhost", "::1"}
    for method in ("start", "stop", "start_server", "stop_server", "send_full_state", "broadcast_update"):
        assert callable(getattr(server, method))
    assert callable(server.start_when_ready)
    with pytest.raises(ValueError, match="loopback"):
        WebSocketUIServer(host="0.0.0.0")


def test_generated_terminal_handles_every_launcher_update_type(tmp_path, monkeypatch):
    path = create_trading_terminal_html(8765, output_path=tmp_path / "terminal.html")
    html = open(path, encoding="utf-8").read()
    for kind in ("full_state", "orderbook", "price", "position", "performance_metrics",
                 "active_orders", "orders_cancelled", "strategy_fill", "status"):
        assert f"message.type === '{kind}'" in html
    assert "handleCancelledOrders(data)" in html
    assert "total_fees" in html
    assert "Realized P&amp;L" not in html
    assert "feedStatus" in html
    assert "Waiting for fresh Binance depth and trade feeds" in html
    assert "Portfolio paper fill attribution" in html
    assert "Selected instrument public prints" in html
    assert "Selected instrument quotes" in html
    assert "marketTapeBody" in html
    assert "orderBookBody" in html
    assert "handlePublicTrade(data)" in html
    assert "normalizePublicTrade" in html
    assert "renderOrderBook(data, countBookUpdate)" in html
    assert "message.market_trades" in html
    assert "updateFeedReadiness({ book_ready: true })" not in html
    assert "updateFeedReadiness({ trade_ready: true })" not in html
    assert "not paper fills" in html
    assert "fill events" in html
    assert "Total Trades" not in html
    for control in ("start_strategy", "stop_strategy", "reset_strategy"):
        assert control in html
    assert "scheduleReconnect" in html
    assert "}, 2000);" in html


def test_phase3_dashboard_is_primary_and_never_fabricates_paper_data(tmp_path):
    path = create_trading_terminal_html(12345, output_path=tmp_path / "paper.html")
    html = open(path, encoding="utf-8").read()
    assert "CrossVenue Paper Trading Terminal" in html
    assert "PAPER ONLY" in html
    assert "const PAPER_WS_PORT = 12345;" in html
    assert "DEMO ONLY" not in html
    assert "startSimulation" not in html
    assert "Simulation Mode" not in html
    assert "Math.random" not in html
    for event in ("full_state", "status", "orderbook", "trade", "position",
                  "performance_metrics", "active_orders", "orders_cancelled", "strategy_fill",
                  "control_response", "error"):
        assert f"message.type === '{event}'" in html
    assert "Not measured in this runtime" in html
    assert "total_completed_orders" in html
    assert "marketDataReady" in html
    assert 'id="streamStatus"' in html
    assert "updateStreamStatus(message.websocket" in html
    assert 'id="strategyVariables"' in html
    assert "Strategy variables" in html
    assert "aspect-ratio: 1 / 1" in html
    assert 'aria-label="Start paper strategy"' in html
    assert 'aria-label="Stop paper strategy"' in html
    server_source = open("src/runtime/websocket_ui_server.py", encoding="utf-8").read()
    assert '"market_trades": list(self.recent_market_trades)' in server_source
    assert 'if update_type == "trade"' in server_source
    assert '"bids":' in server_source and '"asks":' in server_source


def test_auto_start_waits_for_both_feeds_and_never_restarts_after_stop():
    bridge = make_bridge()
    server = WebSocketUIServer(bridge=bridge)
    bridge.on_orderbook_update([(99.0, 1.0)], [(101.0, 1.0)])
    assert not server.start_when_ready()
    bridge.on_trade(price=100.0, quantity=0.1, side="sell")
    assert server.start_when_ready()
    assert bridge.is_running
    bridge.stop_strategy()
    assert not server.start_when_ready()
    assert not bridge.is_running


def test_uptime_does_not_reset_when_strategy_is_restarted():
    bridge = make_bridge()
    started_at = bridge.start_time
    bridge.start_strategy()
    bridge.stop_strategy()
    assert bridge.start_time == started_at


def test_fee_parameter_updates_remain_consistent_across_account_and_risk_reserve():
    bridge = make_bridge()
    bridge.on_orderbook_update([(99.0, 1.0)], [(101.0, 1.0)])
    bridge._process_strategy_order(make_order("fee-sensitive", qty=0.1, price=100.0), 100.0)
    assert bridge.active_orders
    bridge.update_parameters({"maker_fee": 0.002})
    assert not bridge.active_orders
    assert not bridge.risk_manager.pending_orders
    assert bridge.strategy.maker_fee == pytest.approx(0.002)
    assert bridge.pnl_tracker.maker_fee == pytest.approx(0.002)
    assert bridge.risk_manager.estimated_fee_rate == pytest.approx(0.002)
    with pytest.raises(ValueError, match="maker_fee"):
        bridge.update_parameters({"maker_fee": -0.1})


def test_spot_bridge_rejects_short_enabled_risk_manager():
    strategy = AvellanedaStoikov(strategy_id="paper-test")
    risk = RiskManager(RiskLimits(), allow_short=True)
    with pytest.raises(ValueError, match="spot-only"):
        LiveTradingBridge(strategy, risk)


def test_old_public_trade_print_cannot_fill_a_new_quote(monkeypatch):
    bridge = make_bridge()
    bridge.on_orderbook_update([(99.0, 1.0)], [(101.0, 1.0)])
    bridge._process_strategy_order(make_order("time-guard", qty=0.1, price=100.0), 100.0)
    later = max(bridge.order_timestamps.values()) + 0.01
    monkeypatch.setattr(time, "time", lambda: later)
    bridge.is_running = True
    bridge.on_trade(price=100.0, quantity=0.2, side="sell", event_timestamp=time.time() - 2.0)
    assert bridge.total_fills == 0
    bridge.on_trade(price=100.0, quantity=0.2, side="sell", event_timestamp=time.time())
    assert bridge.total_fills == 1
    bridge.is_running = False


def test_strategy_and_pnl_tracker_reject_invalid_initial_fees():
    for fee_name in ("maker_fee", "taker_fee"):
        for invalid_fee in (float("nan"), float("inf"), -0.001):
            with pytest.raises(ValueError, match=fee_name):
                AvellanedaStoikov(**{fee_name: invalid_fee})


def test_generated_dashboard_carries_styles_without_external_css_dependencies(tmp_path):
    from pathlib import Path

    path = Path(create_trading_terminal_html(12345, output_path=tmp_path / "detached" / "paper.html"))
    html = path.read_text(encoding="utf-8")
    assert '<link rel="stylesheet"' not in html
    assert '@import url("./tokens.css")' not in html
    assert '@import url("./base.css")' not in html
    assert "--color-paper:" in html
    assert "overflow-x: clip" in html
    assert "const PAPER_WS_PORT = 12345;" in html


@pytest.mark.parametrize("stylesheet", ["tokens.css", "base.css", "terminal.css"])
def test_missing_dashboard_stylesheet_fails_before_writing_output(tmp_path, monkeypatch, stylesheet):
    import shutil
    import launch

    root = tmp_path / "isolated-project"
    shutil.copytree(launch.PROJECT_ROOT / "assets", root / "assets")
    (root / "assets" / stylesheet).unlink()
    monkeypatch.setattr(launch, "PROJECT_ROOT", root)
    output = tmp_path / "output" / "paper.html"
    with pytest.raises(ValueError, match="stylesheet"):
        create_trading_terminal_html(12345, output_path=output)
    assert not output.exists()
