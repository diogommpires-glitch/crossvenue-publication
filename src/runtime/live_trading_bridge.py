"""
Paper Trading Bridge - Connects Public Market Data to Local Strategies

This module integrates:
- Public market data (Binance WebSocket)
- Existing strategies (Avellaneda-Stoikov, Naive)
- Risk management (validation before local simulated fills)
- Paper-only order execution
- Event-driven UI updates
- Strategy control (start/stop/reset)
"""

from typing import Optional, Dict, List, Callable, Any
import time
import threading
from dataclasses import replace
from datetime import datetime
import math

from ..orderbook.limit_order_book import LimitOrderBook
from ..orderbook.order import Order, Fill
from ..metrics.pnl import PnLTracker
from ..strategies.avellaneda_stoikov import AvellanedaStoikov
from ..strategies.base import Strategy
from ..utils.types import Side, OrderType, OrderStatus, Price, Quantity, Timestamp
from .risk_manager import RiskManager, RiskLimits
from .risk_analytics import RiskAnalytics


class LiveTradingBridge:
    """
    Bridge between real market data and existing strategy code.

    Responsibilities:
    - Maintain local order book from market data
    - Run strategy on market updates
    - Validate orders through risk manager
    - Simulate local paper fills only
    - Track P&L and state
    - Emit events for UI updates
    - Provide strategy control (start/stop/reset)
    """

    def __init__(
        self,
        strategy: Strategy,
        risk_manager: RiskManager,
        connector=None,
        paper_mode: bool = True,
        starting_cash: float = 10000.0,
        paper_queue_fraction: float = 0.5,
        market_data_stale_after: float = 10.0,
    ):
        if not paper_mode:
            raise ValueError("This bridge supports paper trading only; live order execution is disabled")
        if not math.isfinite(starting_cash) or starting_cash <= 0:
            raise ValueError("starting_cash must be finite and positive")
        if not math.isfinite(paper_queue_fraction) or not 0 <= paper_queue_fraction <= 1:
            raise ValueError("paper_queue_fraction must be in [0, 1]")
        if not math.isfinite(market_data_stale_after) or market_data_stale_after <= 0:
            raise ValueError("market_data_stale_after must be finite and positive")
        if getattr(risk_manager, "allow_short", False):
            raise ValueError("The desktop paper bridge is spot-only; allow_short must be false")

        self.strategy = strategy
        self.risk_manager = risk_manager
        self.risk_manager.estimated_fee_rate = float(strategy.maker_fee)
        self.connector = connector
        self.paper_mode = True
        self.paper_queue_fraction = float(paper_queue_fraction)
        self.market_data_stale_after = float(market_data_stale_after)
        self.starting_cash = float(starting_cash)
        self.position = 0.0
        self.cash = self.starting_cash
        self.pnl_tracker = PnLTracker(
            strategy_id=strategy.strategy_id,
            maker_fee=strategy.maker_fee,
            taker_fee=strategy.taker_fee,
            starting_capital=self.starting_cash,
        )
        self.risk_analytics = RiskAnalytics(starting_equity=self.starting_cash)
        self.trades: List[Dict] = []
        self.active_orders: Dict[str, Order] = {}
        self.orders_lock = threading.RLock()
        self.state_lock = threading.RLock()
        self.strategy_lock = threading.RLock()
        self.paper_fill_lock = threading.RLock()
        self.start_time = time.time()
        self.last_strategy_update = 0.0
        self.strategy_interval = 1.0
        self.total_orders_sent = 0
        self.total_fills = 0
        self.total_traded_notional = 0.0
        self.total_rejections = 0
        self.filled_order_ids = set()
        self.event_callbacks: Dict[str, List[Callable]] = {}
        self.events_lock = threading.Lock()
        self.is_running = False
        self.current_mid_price = 0.0
        self.last_orderbook_time: Optional[float] = None
        self.last_trade_time: Optional[float] = None
        self.pnl_thread: Optional[threading.Thread] = None
        self.stop_pnl_thread = threading.Event()
        self.peak_value = self.starting_cash
        self.pnl_returns: List[float] = []
        self.equity_history: List[float] = [self.starting_cash]
        self.fill_prices: List[float] = []
        self.slippage_bps: List[float] = []
        self.expected_prices: List[float] = []
        self.position_history: List[tuple] = []
        self.order_timestamps: Dict[str, float] = {}
        self.last_equity_observation: Optional[float] = None
        self.last_equity_observation_time: Optional[float] = None
        self.risk_manager.reset(balance=self.starting_cash)



    # ==================== Event System ====================

    def on(self, event: str, callback: Callable) -> None:
        """
        Register an event callback.

        Args:
            event: Event name (orderbook_update, trade, fill, pnl_update, position_change)
            callback: Function to call when event occurs
        """
        with self.events_lock:
            if event not in self.event_callbacks:
                self.event_callbacks[event] = []
            self.event_callbacks[event].append(callback)

    def emit(self, event: str, data: Any) -> None:
        """
        Emit an event to all registered callbacks.

        Args:
            event: Event name
            data: Event payload
        """
        with self.events_lock:
            callbacks = self.event_callbacks.get(event, []).copy()

        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                print(f"[ERROR] Event callback failed for '{event}': {e}")

    # ==================== Control Methods ====================

    def start_strategy(self) -> None:
        """Start the trading strategy."""
        if self.is_running:
            print("[CONTROL] Strategy already running")
            return

        self.is_running = True
        self.last_strategy_update = 0.0

        # Start PnL update thread
        self.stop_pnl_thread.clear()
        self.pnl_thread = threading.Thread(target=self._pnl_update_loop, daemon=True)
        self.pnl_thread.start()

        print("[CONTROL] Strategy started")

    def stop_strategy(self) -> None:
        """Stop the trading strategy."""
        if not self.is_running:
            print("[CONTROL] Strategy already stopped")
            return

        self.is_running = False

        # Stop PnL update thread
        self.stop_pnl_thread.set()
        if self.pnl_thread:
            self.pnl_thread.join(timeout=2.0)

        # Cancel all active orders
        self._cancel_all_quotes()

        print("[CONTROL] Strategy stopped")

    def reset_strategy(self) -> None:
        """Reset the paper account, strategy, risk state, and rolling metrics."""
        was_running = self.is_running
        if was_running:
            self.stop_strategy()
        with self.state_lock:
            self.pnl_tracker.reset()
            self.strategy.reset()
            self.position = 0.0
            self.cash = self.starting_cash
            self.trades.clear()
            self.total_orders_sent = 0
            self.total_fills = 0
            self.total_traded_notional = 0.0
            self.total_rejections = 0
            self.filled_order_ids.clear()
            self.current_mid_price = 0.0
            self.last_orderbook_time = None
            self.last_trade_time = None
            self.peak_value = self.starting_cash
            self.pnl_returns.clear()
            self.equity_history = [self.starting_cash]
            self.fill_prices.clear()
            self.slippage_bps.clear()
            self.expected_prices.clear()
            self.position_history.clear()
            self.last_equity_observation = None
            self.last_equity_observation_time = None
            self.start_time = time.time()
            self.risk_analytics.reset()
            self.risk_manager.reset(balance=self.starting_cash)
        with self.orders_lock:
            self.active_orders.clear()
            self.order_timestamps.clear()
        if was_running:
            self.start_strategy()


    def update_parameters(self, params: dict) -> None:
        """Update strategy and cadence parameters without rebasing account equity."""
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")
        if "starting_cash" in params:
            raise ValueError("Starting cash is fixed for an account session; use reset_strategy to restart")
        interval = None
        if "strategy_interval" in params:
            interval = float(params["strategy_interval"])
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError("strategy_interval must be finite and positive")
        strategy_params = {key: value for key, value in params.items() if key != "strategy_interval"}
        fee_changed = any(name in strategy_params for name in ("maker_fee", "taker_fee"))
        for fee_name in ("maker_fee", "taker_fee"):
            if fee_name in strategy_params:
                try:
                    fee_rate = float(strategy_params[fee_name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{fee_name} must be finite and between 0 and 0.1") from exc
                if not math.isfinite(fee_rate) or not 0.0 <= fee_rate <= 0.1:
                    raise ValueError(f"{fee_name} must be finite and between 0 and 0.1")
                strategy_params[fee_name] = fee_rate
        updater = getattr(self.strategy, "update_parameters", None)
        if strategy_params and updater is None:
            unknown = [key for key in strategy_params if not hasattr(self.strategy, key)]
            if unknown:
                raise ValueError(f"Unsupported strategy parameters: {', '.join(unknown)}")

        with self.strategy_lock:
            if strategy_params:
                if updater is None:
                    for key, value in strategy_params.items():
                        setattr(self.strategy, key, value)
                else:
                    updater(strategy_params)
                self.pnl_tracker.maker_fee = float(self.strategy.maker_fee)
                self.pnl_tracker.taker_fee = float(self.strategy.taker_fee)
                self.risk_manager.estimated_fee_rate = float(self.strategy.maker_fee)
                if fee_changed:
                    self._cancel_all_quotes()
            if interval is not None:
                self.strategy_interval = interval
        updated = []
        if interval is not None:
            updated.append(f"strategy_interval={interval}")
        if strategy_params:
            updated.extend(f"{key}={value}" for key, value in strategy_params.items())
        if updated:
            print(f"[CONTROL] Parameters updated: {', '.join(updated)}")


    def _pnl_update_loop(self) -> None:
        """Publish marked equity and cancel quotes whenever public data goes stale."""
        while not self.stop_pnl_thread.is_set():
            if self.is_running:
                if not self.market_data_is_fresh():
                    self._cancel_all_quotes()
                else:
                    pnl_data = self._calculate_pnl(record_observation=True)
                    self.emit("pnl_update", pnl_data)
                    self.emit("performance_metrics", self.get_performance_metrics())
                    self.emit("active_orders", {"orders": self.get_active_orders_detail()})
            self.stop_pnl_thread.wait(1.0)



    def _calculate_pnl(self, record_observation: bool = False) -> Dict:
        """Return net mark-to-market P&L and optionally record a 1-second sample."""
        with self.state_lock:
            mark_price = self.current_mid_price
            current = self.pnl_tracker.get_current_pnl(mark_price)
            equity = self.starting_cash + current["total_pnl"]
            self.position = float(self.pnl_tracker.inventory)
            self.cash = self.starting_cash + self.pnl_tracker.cash
            self.peak_value = max(self.peak_value, equity)

            if record_observation and mark_price > 0:
                now = time.time()
                if self.last_equity_observation is not None and equity > 0:
                    self.pnl_returns.append((equity - self.last_equity_observation) / self.last_equity_observation)
                    if len(self.pnl_returns) > 1000:
                        del self.pnl_returns[:-1000]
                self.last_equity_observation = equity
                self.last_equity_observation_time = now
                self.equity_history.append(equity)
                if len(self.equity_history) > 10000:
                    del self.equity_history[:-10000]
                self.risk_analytics.update(
                    pnl=current["total_pnl"],
                    equity=equity,
                    position=self.position,
                    timestamp=now,
                    current_price=mark_price,
                )
            return {
                "timestamp": datetime.now().isoformat(),
                "position": self.position,
                "cash": self.cash,
                "mid_price": mark_price,
                "equity": equity,
                "realized_pnl": current["realized_pnl"],
                "unrealized_pnl": current["unrealized_pnl"],
                "total_pnl": current["total_pnl"],
                "total_fees": self.pnl_tracker.fees_paid,
                "return_pct": current["total_pnl"] / self.starting_cash * 100.0,
            }


    # ==================== Market Data Handlers ====================

    def on_orderbook_update(self, bids: List[tuple], asks: List[tuple]):
        """Update the live public market mark and schedule decisions only on fresh data."""
        if not bids or not asks:
            return
        best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
        if not math.isfinite(best_bid) or not math.isfinite(best_ask) or best_bid <= 0 or best_ask < best_bid:
            return
        mid_price = (best_bid + best_ask) / 2.0
        now_wall = time.time()
        current_time = Timestamp(now_wall - self.start_time)
        with self.state_lock:
            self.current_mid_price = mid_price
            self.last_orderbook_time = now_wall
            self.risk_manager.update_state(
                position=self.position,
                balance=self.starting_cash + self.pnl_tracker.get_current_pnl(mid_price)["total_pnl"],
                starting_balance=self.starting_cash,
                cash_balance=self.cash,
            )
        self.emit("orderbook_update", {
            "timestamp": datetime.now().isoformat(), "bids": bids[:5], "asks": asks[:5],
            "best_bid": best_bid, "best_ask": best_ask, "mid_price": mid_price,
            "spread": best_ask - best_bid,
        })
        if self.risk_manager.circuit_breaker_active:
            self._cancel_all_quotes()
            self.emit("risk_circuit_breaker", self.risk_manager.get_status())
            return
        if self.is_running and self.market_data_is_fresh():
            now = time.monotonic()
            if now - self.last_strategy_update >= self.strategy_interval:
                with self.strategy_lock:
                    now = time.monotonic()
                    if self.is_running and now - self.last_strategy_update >= self.strategy_interval:
                        self._run_strategy_update(mid_price, best_bid, best_ask, current_time)
                        self.last_strategy_update = now



    def on_trade(self, price: float, quantity: float, side: str, event_timestamp: Optional[float] = None):
        """Use Binance public prints as paper-fill evidence only."""
        if not all(math.isfinite(value) for value in (price, quantity)) or price <= 0 or quantity <= 0:
            return
        if side not in {"buy", "sell"}:
            return
        received_at = time.time()
        if event_timestamp is None:
            trade_time = None
        else:
            trade_time = float(event_timestamp)
            if (not math.isfinite(trade_time)
                    or trade_time > received_at + 1.0
                    or received_at - trade_time > self.market_data_stale_after + 1.0):
                return
        self.last_trade_time = received_at
        self.emit("trade", {
            "timestamp": datetime.now().isoformat(), "price": float(price),
            "quantity": float(quantity), "side": side, "event_timestamp": trade_time,
        })
        if not self.market_data_is_fresh():
            self._cancel_all_quotes()
            return
        self._check_paper_fills(float(price), side, float(quantity), trade_time)



    def _run_strategy_update(self, mid_price: float, best_bid: float, best_ask: float, timestamp: Timestamp):
        """Run strategy under its state lock using fresh Binance market data."""
        if not self.market_data_is_fresh() or self.risk_manager.circuit_breaker_active:
            self._cancel_all_quotes()
            return
        with self.state_lock:
            pnl = self.pnl_tracker.get_current_pnl(mid_price)
            self.position = float(self.pnl_tracker.inventory)
            self.cash = self.starting_cash + self.pnl_tracker.cash
            self.risk_manager.update_state(
                position=self.position,
                balance=self.starting_cash + pnl["total_pnl"],
                starting_balance=self.starting_cash,
                cash_balance=self.cash,
            )
        if self.risk_manager.circuit_breaker_active:
            self._cancel_all_quotes()
            return
        self._cancel_all_quotes()
        market_state = {
            "time": timestamp, "mid_price": mid_price, "best_bid": best_bid,
            "best_ask": best_ask, "spread": best_ask - best_bid,
        }
        for order in self.strategy.on_market_event(market_state):
            self._process_strategy_order(order, mid_price)



    def _process_strategy_order(self, order: Order, current_price: float):
        """Atomically validate and reserve a paper limit order before publishing it."""
        if not self.paper_mode or order.order_type != OrderType.LIMIT or order.price is None:
            self.total_rejections += 1
            return
        side = "buy" if order.side == Side.BUY else "sell"
        quantity, price = float(order.quantity), float(order.price)
        with self.strategy_lock:
            with self.orders_lock:
                if order.order_id in self.active_orders:
                    return
                is_valid, violation, message = self.risk_manager.reserve_order(
                    order.order_id, side, quantity, price
                )
                if not is_valid:
                    self.total_rejections += 1
                    rejection = {"side": side, "quantity": quantity, "price": price,
                                 "reason": message, "violation": violation.value if violation else None}
                else:
                    self.active_orders[order.order_id] = order
                    self.order_timestamps[order.order_id] = time.time()
                    rejection = None
        if rejection is not None:
            self.emit("order_rejected", rejection)
            return
        self.total_orders_sent += 1
        self.emit("order_accepted", {"order_id": order.order_id, "side": side,
                                     "quantity": quantity, "price": price})



    def _execute_paper_order(self, order: Order):
        """Compatibility entry point; order validation is performed before this call."""
        self._process_strategy_order(order, self.current_mid_price)


    def _execute_real_order(self, order: Order):
        """Live exchange order submission is deliberately unsupported."""
        raise NotImplementedError("This project is configured for paper execution only")


    def _check_paper_fills(self, trade_price: float, trade_side: str, trade_volume: float,
                           trade_timestamp: Optional[float] = None):
        """Share one public print across queue-ahead estimates without exceeding its volume."""
        if (not self.paper_mode or not self.is_running or self.risk_manager.circuit_breaker_active
                or not self.market_data_is_fresh() or not math.isfinite(trade_volume)
                or trade_volume <= 0 or trade_timestamp is None
                or not math.isfinite(trade_timestamp)):
            if self.risk_manager.circuit_breaker_active or not self.market_data_is_fresh():
                self._cancel_all_quotes()
            return
        available_volume = trade_volume * self.paper_queue_fraction
        if available_volume <= 1e-12:
            return
        with self.strategy_lock:
            with self.paper_fill_lock:
                with self.orders_lock:
                    eligible = []
                    for order in self.active_orders.values():
                        accepted_at = self.order_timestamps.get(order.order_id)
                        if accepted_at is None or trade_timestamp <= accepted_at:
                            continue
                        crosses = (
                            order.side == Side.BUY and trade_side == "sell"
                            and order.price is not None and trade_price <= float(order.price)
                        ) or (
                            order.side == Side.SELL and trade_side == "buy"
                            and order.price is not None and trade_price >= float(order.price)
                        )
                        if crosses:
                            eligible.append(order)
                    eligible.sort(key=lambda order: (
                        -float(order.price) if order.side == Side.BUY else float(order.price),
                        float(order.timestamp), order.order_id,
                    ))
                filled_order_ids = []
                for order in eligible:
                    if available_volume <= 1e-12 or not self.is_running or self.risk_manager.circuit_breaker_active:
                        break
                    with self.orders_lock:
                        live_order = self.active_orders.get(order.order_id)
                        if live_order is None or live_order.price is None:
                            continue
                        fill_price = float(live_order.price)
                        quantity = min(float(live_order.quantity), available_volume)
                    self._record_fill(live_order, fill_price, quantity)
                    available_volume -= quantity
                    remaining = float(live_order.quantity) - quantity
                    with self.orders_lock:
                        if order.order_id not in self.active_orders:
                            continue
                        if remaining <= 1e-12:
                            self.active_orders.pop(order.order_id, None)
                            self.order_timestamps.pop(order.order_id, None)
                            self.risk_manager.clear_pending_order(order.order_id)
                            self.filled_order_ids.add(order.order_id)
                            filled_order_ids.append(order.order_id)
                        else:
                            self.active_orders[order.order_id] = replace(
                                self.active_orders[order.order_id], quantity=Quantity(remaining),
                                status=OrderStatus.PARTIALLY_FILLED,
                            )
                            self.risk_manager.update_pending_order(order.order_id, remaining)
                if self.risk_manager.circuit_breaker_active:
                    self._cancel_all_quotes()
                if filled_order_ids:
                    self.emit("orders_filled", {"order_ids": filled_order_ids})



    def _record_fill(self, order: Order, fill_price: float, fill_quantity: float = None):
        """Book a maker fill at the execution price and mark it at the current public mid."""
        quantity = float(order.quantity) if fill_quantity is None else float(fill_quantity)
        if quantity <= 0 or not math.isfinite(quantity) or not math.isfinite(fill_price) or fill_price <= 0:
            raise ValueError("fill quantity and price must be finite and positive")
        timestamp = Timestamp(time.time() - self.start_time)
        fill = Fill(order_id=order.order_id, side=order.side, quantity=Quantity(quantity),
                    price=Price(fill_price), timestamp=timestamp,
                    trader_id=self.strategy.strategy_id, is_aggressor=False)
        with self.strategy_lock:
            with self.state_lock:
                mark_price = self.current_mid_price if self.current_mid_price > 0 else float(fill_price)
                before_realized = self.pnl_tracker.realized_pnl
                self.pnl_tracker.record_fill(fill, mark_price, timestamp, mark_price)
                self.strategy.on_fill(fill, fill_price)
                self.position = float(self.pnl_tracker.inventory)
                self.cash = self.starting_cash + self.pnl_tracker.cash
                current = self.pnl_tracker.get_current_pnl(mark_price)
                self.risk_manager.update_state(
                    position=self.position, balance=self.starting_cash + current["total_pnl"],
                    starting_balance=self.starting_cash, cash_balance=self.cash,
                )
                realized_delta = self.pnl_tracker.realized_pnl - before_realized
                fee = quantity * fill_price * self.strategy.maker_fee
                trade = {
                    "timestamp": datetime.now().isoformat(), "order_id": order.order_id,
                    "side": order.side.name, "quantity": quantity, "price": float(fill_price),
                    "position_after": self.position, "cash_after": self.cash,
                    "realized_pnl": realized_delta, "fill_fee": fee,
                    "total_pnl": current["total_pnl"],
                }
                self.trades.append(trade)
                self.total_fills += 1
                self.total_traded_notional += quantity * fill_price
                self.fill_prices.append(float(fill_price))
                expected = float(order.price) if order.price else mark_price
                if expected > 0:
                    direction = 1.0 if order.side == Side.BUY else -1.0
                    self.slippage_bps.append(direction * (float(fill_price) - expected) / expected * 10000.0)
                    self.slippage_bps = self.slippage_bps[-100:]
                self.expected_prices.append(float(order.price) if order.price else mark_price)
                self.position_history.append((time.time(), self.position))
                self.position_history = self.position_history[-200:]
                self.fill_prices = self.fill_prices[-100:]
            self.emit("fill", {"timestamp": trade["timestamp"], "order_id": order.order_id,
                               "side": order.side.name, "quantity": quantity, "price": float(fill_price),
                               "position": self.position, "cash": self.cash,
                               "realized_pnl": realized_delta, "total_pnl": current["total_pnl"],
                               "fee": trade["fill_fee"]})
            self.emit("position_change", {"position": self.position, "cash": self.cash})



    def _cancel_all_quotes(self):
        """Serialize cancellation with fills and clear every pending risk reservation."""
        with self.strategy_lock:
            with self.paper_fill_lock:
                with self.orders_lock:
                    cancelled_ids = list(self.active_orders)
                    self.active_orders.clear()
                    self.order_timestamps.clear()
                    for order_id in cancelled_ids:
                        self.risk_manager.clear_pending_order(order_id)
        if cancelled_ids:
            self.emit("orders_cancelled", {"order_ids": cancelled_ids})



    def market_data_is_fresh(self, now: Optional[float] = None) -> bool:
        """Whether both public depth and trade streams are fresh enough for paper fills."""
        current = time.time() if now is None else float(now)
        return self.book_data_is_fresh(current) and self.trade_data_is_fresh(current)

    def book_data_is_fresh(self, now: Optional[float] = None) -> bool:
        current = time.time() if now is None else float(now)
        return (self.last_orderbook_time is not None
                and 0 <= current - self.last_orderbook_time <= self.market_data_stale_after)

    def trade_data_is_fresh(self, now: Optional[float] = None) -> bool:
        current = time.time() if now is None else float(now)
        return (self.last_trade_time is not None
                and 0 <= current - self.last_trade_time <= self.market_data_stale_after)

    def get_status(self) -> Dict:
        """Return account status and explicit paper/feed readiness fields."""
        pnl = self._calculate_pnl()
        with self.orders_lock:
            active_count = len(self.active_orders)
            completed_order_count = len(self.filled_order_ids)
        now = time.time()
        book_age = None if self.last_orderbook_time is None else max(0.0, now - self.last_orderbook_time)
        trade_age = None if self.last_trade_time is None else max(0.0, now - self.last_trade_time)
        book_ready = self.book_data_is_fresh(now)
        trade_ready = self.trade_data_is_fresh(now)
        return {
            "uptime_seconds": time.time() - self.start_time, "position": pnl["position"],
            "cash": pnl["cash"], "mid_price": pnl["mid_price"], "equity": pnl["equity"],
            "realized_pnl": pnl["realized_pnl"], "unrealized_pnl": pnl["unrealized_pnl"],
            "total_pnl": pnl["total_pnl"], "total_fees": pnl["total_fees"],
            "return_pct": pnl["return_pct"], "total_trades": len(self.trades),
            "total_orders_sent": self.total_orders_sent, "total_fills": self.total_fills,
            "total_completed_orders": completed_order_count,
            "total_traded_notional": self.total_traded_notional,
            "total_rejections": self.total_rejections, "active_orders": active_count,
            "paper_mode": True, "market_data_source": "Binance public market data",
            "execution_mode": "paper simulation", "is_running": self.is_running,
            "book_ready": book_ready,
            "trade_ready": trade_ready,
            "market_data_ready": book_ready and trade_ready,
            "orderbook_age_seconds": book_age, "trade_age_seconds": trade_age,
            "circuit_breaker_active": self.risk_manager.circuit_breaker_active,
        }



    def get_full_status(self) -> Dict:
        """Return a consistent dashboard snapshot with marked equity and fees."""
        pnl = self._calculate_pnl()
        with self.orders_lock:
            active_orders = [
                {
                    "order_id": order.order_id,
                    "side": order.side.name,
                    "price": float(order.price) if order.price else None,
                    "quantity": float(order.quantity),
                    "order_type": order.order_type.name,
                    "age_seconds": max(0.0, time.time() - self.order_timestamps.get(order.order_id, time.time())),
                }
                for order in self.active_orders.values()
            ]
        limit = self.risk_manager.limits
        risk_status = self.risk_manager.get_status()
        metrics = self.risk_analytics.get_risk_metrics(pnl["mid_price"] if pnl["mid_price"] > 0 else None)
        with self.orders_lock:
            completed_order_count = len(self.filled_order_ids)
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "is_running": self.is_running,
            "paper_mode": True,
            "market_data_source": "Binance public market data",
            "execution_mode": "paper simulation",
            "mid_price": pnl["mid_price"],
            "position": pnl["position"],
            "cash": pnl["cash"],
            "equity": pnl["equity"],
            "starting_cash": self.starting_cash,
            "realized_pnl": pnl["realized_pnl"],
            "unrealized_pnl": pnl["unrealized_pnl"],
            "total_pnl": pnl["total_pnl"],
            "total_fees": pnl["total_fees"],
            "return_pct": pnl["return_pct"],
            "total_orders_sent": self.total_orders_sent,
            "total_fills": self.total_fills,
            "total_completed_orders": completed_order_count,
            "total_traded_notional": self.total_traded_notional,
            "total_rejections": self.total_rejections,
            "fill_rate": self.calculate_fill_rate(),
            "active_orders": active_orders,
            "active_order_count": len(active_orders),
            "recent_trades": self.trades[-20:],
            "total_trades": len(self.trades),
            "risk_metrics": {
                "position_limit": limit.max_position_size,
                "position_usage_pct": abs(pnl["position"]) / limit.max_position_size * 100.0,
                "daily_loss_limit": limit.max_daily_loss,
                "current_drawdown": max(0.0, self.peak_value - pnl["equity"]),
                "circuit_breaker_active": risk_status["circuit_breaker_active"],
                "pending_orders_count": risk_status["pending_orders_count"],
            },
            "performance_metrics": {
                "sharpe_ratio_per_second": self.calculate_sharpe_ratio(),
                "max_drawdown": self.calculate_max_drawdown(),
                "fill_rate": self.calculate_fill_rate(),
                "avg_slippage_bps": self.calculate_avg_slippage(),
                "var_95": metrics.var_95,
                "cvar_95": metrics.cvar_95,
                "volatility_per_observation": metrics.volatility,
                "position_notional": metrics.position_notional,
            },
            "book_ready": self.book_data_is_fresh(),
            "trade_ready": self.trade_data_is_fresh(),
            "market_data_ready": self.market_data_is_fresh(),
            "orderbook_age_seconds": max(0.0, time.time() - self.last_orderbook_time) if self.last_orderbook_time else None,
            "trade_age_seconds": max(0.0, time.time() - self.last_trade_time) if self.last_trade_time else None,
            "account_type": "spot paper account",
            "strategy_interval": self.strategy_interval,
            "strategy_params": self._get_strategy_params(),
        }


    def _get_strategy_params(self) -> Dict:
        """Extract strategy parameters for display."""
        params = {}

        # Common parameters to check
        param_names = ["risk_aversion", "order_size", "gamma", "k", "sigma", "T",
                      "bid_offset", "ask_offset", "quote_size", "inventory_target"]

        for name in param_names:
            if hasattr(self.strategy, name):
                value = getattr(self.strategy, name)
                params[name] = float(value) if hasattr(value, '__float__') else value

        return params

    # ==================== Phase 3: Performance Metrics ====================

    def calculate_sharpe_ratio(self, lookback: int = 100) -> float:
        """Return non-annualized Sharpe over the latest one-second equity samples."""
        if len(self.pnl_returns) < 2:
            return 0.0
        import statistics
        returns = self.pnl_returns[-lookback:]
        deviation = statistics.stdev(returns)
        if deviation <= 1e-15:
            return 0.0
        return statistics.mean(returns) / deviation


    def calculate_max_drawdown(self) -> float:
        """Return the largest peak-to-trough marked-equity loss in quote currency."""
        if len(self.equity_history) < 2:
            return 0.0
        peak = self.equity_history[0]
        largest = 0.0
        for equity in self.equity_history[1:]:
            peak = max(peak, equity)
            largest = max(largest, peak - equity)
        current_equity = self._calculate_pnl()["equity"]
        return max(largest, peak - current_equity)


    def calculate_fill_rate(self) -> float:
        """Percentage of submitted orders that received at least one fill."""
        if self.total_orders_sent == 0:
            return 0.0
        return min(100.0, len(self.filled_order_ids) / self.total_orders_sent * 100.0)


    def calculate_avg_slippage(self) -> float:
        """Return average signed maker-fill slippage in basis points (negative is favorable)."""
        if not self.slippage_bps:
            return 0.0
        import statistics
        return statistics.mean(self.slippage_bps)


    def get_performance_metrics(self) -> Dict:
        """Return metrics with explicit observation cadence and currency units."""
        return {
            "sharpe_ratio_per_second": self.calculate_sharpe_ratio(),
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "max_drawdown": self.calculate_max_drawdown(),
            "fill_rate": self.calculate_fill_rate(),
            "avg_slippage_bps": self.calculate_avg_slippage(),
            "avg_slippage": self.calculate_avg_slippage(),
            "total_fees": self.pnl_tracker.fees_paid,
        }


    def get_active_orders_detail(self) -> List[Dict]:
        """
        Get detailed active orders with timestamps.

        Returns:
            List of active order details
        """
        with self.orders_lock:
            current_time = time.time()
            orders = []
            for order_id, order in self.active_orders.items():
                order_time = self.order_timestamps.get(order_id, current_time)
                age_seconds = current_time - order_time

                orders.append({
                    "order_id": order_id,
                    "side": order.side.name,
                    "price": float(order.price) if order.price else None,
                    "size": float(order.quantity),
                    "age_seconds": age_seconds,
                    "age": self._format_duration(age_seconds),
                })

            return orders

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m"
        else:
            return f"{int(seconds / 3600)}h"
