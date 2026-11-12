"""Pre-trade controls for the Binance-data paper-trading bridge.

`balance` is marked-to-market account equity. `cash_balance` is the spendable
quote-currency balance. Orders remain paper-only in the desktop launcher.
"""

from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from typing import Dict, Optional, Tuple, Union


class RiskViolation(Enum):
    MAX_POSITION = "max_position_exceeded"
    MAX_ORDER_SIZE = "max_order_size_exceeded"
    MAX_DAILY_LOSS = "max_daily_loss_exceeded"
    MAX_TOTAL_EXPOSURE = "max_total_exposure_exceeded"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    RATE_LIMIT = "rate_limit_exceeded"
    CIRCUIT_BREAKER = "circuit_breaker_triggered"
    INSUFFICIENT_INVENTORY = "insufficient_base_asset_inventory"


@dataclass
class RiskLimits:
    """Limits in BTC, USDT, and orders per rolling minute."""

    max_position_size: float = 1.0
    max_order_size: float = 0.1
    max_daily_loss: float = 1000.0
    max_total_exposure: float = 10000.0
    max_orders_per_minute: int = 10
    circuit_breaker_loss_pct: float = 0.05
    min_balance_reserve: float = 100.0

    def __post_init__(self) -> None:
        positive = {
            "max_position_size": self.max_position_size,
            "max_order_size": self.max_order_size,
            "max_total_exposure": self.max_total_exposure,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.max_daily_loss) or self.max_daily_loss < 0:
            raise ValueError("max_daily_loss must be finite and non-negative")
        if not math.isfinite(self.circuit_breaker_loss_pct) or not 0 <= self.circuit_breaker_loss_pct <= 1:
            raise ValueError("circuit_breaker_loss_pct must be between 0 and 1")
        if not math.isfinite(self.min_balance_reserve) or self.min_balance_reserve < 0:
            raise ValueError("min_balance_reserve must be finite and non-negative")
        if not isinstance(self.max_orders_per_minute, int) or self.max_orders_per_minute < 1:
            raise ValueError("max_orders_per_minute must be a positive integer")


class RiskManager:
    """Validate each proposed order against equity, cash, and pending exposure."""

    def __init__(self, limits: RiskLimits, allow_short: bool = False):
        self.limits = limits
        self.allow_short = bool(allow_short)
        self.current_position = 0.0
        self.starting_balance = 0.0
        self.current_balance = 0.0
        self.cash_balance = 0.0
        self.daily_pnl = 0.0
        self.pending_orders: Dict[str, Dict[str, Union[float, str]]] = {}
        self.order_timestamps = []
        self.circuit_breaker_active = False
        self.circuit_breaker_time: Optional[float] = None
        self.violations_count: Dict[RiskViolation, int] = {}
        self._lock = threading.RLock()
        self.estimated_fee_rate = 0.001

    def update_state(
        self,
        position: float,
        balance: float,
        starting_balance: float,
        cash_balance: Optional[float] = None,
    ) -> None:
        """Update mark-to-market equity and available quote currency."""
        values = (position, balance, starting_balance)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("position and balances must be finite")
        if starting_balance <= 0:
            raise ValueError("starting_balance must be positive")
        spendable_cash = balance if cash_balance is None else cash_balance
        if not math.isfinite(spendable_cash):
            raise ValueError("cash_balance must be finite")

        with self._lock:
            self.current_position = float(position)
            self.current_balance = float(balance)
            self.cash_balance = float(spendable_cash)
            if self.starting_balance == 0:
                self.starting_balance = float(starting_balance)
            self.daily_pnl = self.current_balance - self.starting_balance
            loss_fraction = max(0.0, -self.daily_pnl) / self.starting_balance
            if loss_fraction >= self.limits.circuit_breaker_loss_pct and self.daily_pnl < 0:
                self._trigger_circuit_breaker()

    def validate_order(self, side: str, quantity: float, price: float) -> Tuple[bool, Optional[RiskViolation], str]:
        """Return whether a proposed order passes all configured controls.

        Pending orders are treated as independently fillable. This checks the
        long and short extremes separately instead of netting opposing quotes.
        """
        if not isinstance(side, str) or side.lower() not in {"buy", "sell"}:
            return False, None, "side must be 'buy' or 'sell'"
        side = side.lower()
        if not math.isfinite(quantity) or quantity <= 0:
            return False, None, "quantity must be finite and positive"
        if not math.isfinite(price) or price <= 0:
            return False, None, "price must be finite and positive"

        with self._lock:
            if self.circuit_breaker_active:
                return False, RiskViolation.CIRCUIT_BREAKER, "Circuit breaker active"
            if not self._check_rate_limit():
                return self._reject(RiskViolation.RATE_LIMIT, "Order rate limit exceeded")
            if quantity > self.limits.max_order_size + 1e-12:
                return self._reject(RiskViolation.MAX_ORDER_SIZE, "Order exceeds maximum order size")
            if self.daily_pnl < -self.limits.max_daily_loss:
                return self._reject(RiskViolation.MAX_DAILY_LOSS, "Session loss limit exceeded")

            pending_buys = sum(float(order["quantity"]) for order in self.pending_orders.values() if order["side"] == "buy")
            pending_sells = sum(float(order["quantity"]) for order in self.pending_orders.values() if order["side"] == "sell")
            if side == "sell" and not self.allow_short:
                available_inventory = max(0.0, self.current_position - pending_sells)
                if quantity > available_inventory + 1e-12:
                    return self._reject(RiskViolation.INSUFFICIENT_INVENTORY, "Sell order exceeds unreserved base-asset inventory")
            pending_buy_notional = sum(
                float(order["quantity"]) * float(order["price"]) * (1.0 + self.estimated_fee_rate)
                for order in self.pending_orders.values()
                if order["side"] == "buy"
            )

            candidate_buy = quantity if side == "buy" else 0.0
            candidate_sell = quantity if side == "sell" else 0.0
            worst_long = self.current_position + pending_buys + candidate_buy
            worst_short = self.current_position - pending_sells - candidate_sell
            worst_position = max(abs(worst_long), abs(worst_short), abs(self.current_position))
            if worst_position > self.limits.max_position_size + 1e-12:
                return self._reject(RiskViolation.MAX_POSITION, "Potential position exceeds configured limit")

            if worst_position * price > self.limits.max_total_exposure + 1e-9:
                return self._reject(RiskViolation.MAX_TOTAL_EXPOSURE, "Potential marked exposure exceeds configured limit")

            if side == "buy":
                required = pending_buy_notional + quantity * price * (1.0 + self.estimated_fee_rate) + self.limits.min_balance_reserve
                if required > self.cash_balance + 1e-9:
                    return self._reject(RiskViolation.INSUFFICIENT_BALANCE, "Insufficient unreserved quote balance")

            return True, None, "Order validated"

    def reserve_order(self, order_id: str, side: str, quantity: float, price: float):
        """Atomically validate an order, record its rate-limit slot, and reserve exposure."""
        with self._lock:
            if order_id in self.pending_orders:
                return False, None, "Order ID already has a pending reservation"
            valid, violation, message = self.validate_order(side, quantity, price)
            if not valid:
                return valid, violation, message
            normalized_side = side.lower()
            self.pending_orders[order_id] = {
                "side": normalized_side, "quantity": float(quantity), "price": float(price)
            }
            now = time.time()
            self.order_timestamps = [stamp for stamp in self.order_timestamps if stamp > now - 60.0]
            self.order_timestamps.append(now)
            return True, None, "Order validated and reserved"

    def register_pending_order(self, order_id: str, side: str, quantity: float, price: float = 0.0) -> None:
        """Reserve risk and cash for an accepted resting order."""
        normalized_side = side.lower()
        if normalized_side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        if not math.isfinite(quantity) or quantity <= 0:
            raise ValueError("quantity must be finite and positive")
        if not math.isfinite(price) or price < 0:
            raise ValueError("price must be finite and non-negative")
        with self._lock:
            self.pending_orders[order_id] = {
                "side": normalized_side,
                "quantity": float(quantity),
                "price": float(price),
            }

    def update_pending_order(self, order_id: str, quantity: float, price: Optional[float] = None) -> None:
        """Replace the remaining risk reservation after a partial fill."""
        if not math.isfinite(quantity) or quantity < 0:
            raise ValueError("remaining quantity must be finite and non-negative")
        with self._lock:
            order = self.pending_orders.get(order_id)
            if order is None:
                return
            if quantity <= 1e-12:
                self.pending_orders.pop(order_id, None)
                return
            order["quantity"] = float(quantity)
            if price is not None:
                if not math.isfinite(price) or price < 0:
                    raise ValueError("price must be finite and non-negative")
                order["price"] = float(price)

    def clear_pending_order(self, order_id: str) -> None:
        with self._lock:
            self.pending_orders.pop(order_id, None)

    def clear_pending_orders(self) -> None:
        with self._lock:
            self.pending_orders.clear()

    def get_pending_exposure(self) -> float:
        with self._lock:
            return sum(
                float(order["quantity"]) * (1.0 if order["side"] == "buy" else -1.0)
                for order in self.pending_orders.values()
            )

    def record_order(self) -> None:
        with self._lock:
            now = time.time()
            cutoff = now - 60.0
            self.order_timestamps = [stamp for stamp in self.order_timestamps if stamp > cutoff]
            self.order_timestamps.append(now)

    def reset(self, balance: float, position: float = 0.0) -> None:
        """Reset session controls to a clean paper-trading account state."""
        if not math.isfinite(balance) or balance <= 0 or not math.isfinite(position):
            raise ValueError("reset requires positive finite balance and finite position")
        with self._lock:
            self.current_position = float(position)
            self.starting_balance = float(balance)
            self.current_balance = float(balance)
            self.cash_balance = float(balance)
            self.daily_pnl = 0.0
            self.pending_orders.clear()
            self.order_timestamps.clear()
            self.circuit_breaker_active = False
            self.circuit_breaker_time = None
            self.violations_count.clear()

    def reset_circuit_breaker(self) -> None:
        with self._lock:
            self.circuit_breaker_active = False
            self.circuit_breaker_time = None

    def get_status(self) -> Dict:
        with self._lock:
            now = time.time()
            recent = [stamp for stamp in self.order_timestamps if stamp > now - 60.0]
            pending_buys = sum(float(order["quantity"]) for order in self.pending_orders.values() if order["side"] == "buy")
            pending_sells = sum(float(order["quantity"]) for order in self.pending_orders.values() if order["side"] == "sell")
            worst_abs = max(
                abs(self.current_position + pending_buys),
                abs(self.current_position - pending_sells),
                abs(self.current_position),
            )
            return {
                "position": self.current_position,
                "pending_exposure": self.get_pending_exposure(),
                "worst_case_position": worst_abs,
                "balance": self.current_balance,
                "cash_balance": self.cash_balance,
                "daily_pnl": self.daily_pnl,
                "position_utilization": abs(self.current_position) / self.limits.max_position_size,
                "circuit_breaker_active": self.circuit_breaker_active,
                "violations": {key.value: value for key, value in self.violations_count.items()},
                "orders_last_minute": len(recent),
                "pending_orders_count": len(self.pending_orders),
            }

    def _check_rate_limit(self) -> bool:
        cutoff = time.time() - 60.0
        self.order_timestamps = [stamp for stamp in self.order_timestamps if stamp > cutoff]
        return len(self.order_timestamps) < self.limits.max_orders_per_minute

    def _reject(self, violation: RiskViolation, message: str) -> Tuple[bool, RiskViolation, str]:
        self.violations_count[violation] = self.violations_count.get(violation, 0) + 1
        return False, violation, message

    def _trigger_circuit_breaker(self) -> None:
        if not self.circuit_breaker_active:
            self.circuit_breaker_active = True
            self.circuit_breaker_time = time.time()
