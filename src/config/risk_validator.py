"""Risk validation and enforcement for trading system.

Validates orders and positions against configured risk limits,
with circuit breaker support and real-time monitoring.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import logging


class RiskViolation(Exception):
    """Raised when a risk limit is violated."""
    pass


class CircuitBreakerTripped(Exception):
    """Raised when circuit breaker is triggered."""
    pass


class RiskValidator:
    """Validates trading decisions against risk limits."""

    def __init__(self, risk_limits: Dict[str, Any]):
        """Initialize risk validator.

        Args:
            risk_limits: Risk limits configuration dictionary
        """
        self.limits = risk_limits
        self.logger = logging.getLogger(__name__)

        # Track current state
        self.current_position_base = 0.0
        self.current_position_quote = 0.0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0

        # Track losses
        self.trade_losses: List[float] = []
        self.daily_loss_start = datetime.now()
        self.weekly_loss_start = datetime.now()
        self.monthly_loss_start = datetime.now()

        # Track order rate limits
        self.order_timestamps = deque(maxlen=1000)
        self.cancel_timestamps = deque(maxlen=1000)

        # Circuit breaker state
        self.circuit_breaker_active = False
        self.circuit_breaker_time: Optional[datetime] = None
        self.circuit_breaker_reason: Optional[str] = None

        # Consecutive loss tracking
        self.consecutive_losses = 0
        self.last_trade_profitable = True

        self.logger.info("Risk validator initialized with limits")

    def validate_order(
        self,
        side: str,
        size: float,
        price: float,
        current_book_depth: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate order against risk limits.

        Args:
            side: Order side ('buy' or 'sell')
            size: Order size in base currency
            price: Order price
            current_book_depth: Current order book depth in quote currency

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check circuit breaker
        if self.circuit_breaker_active:
            if not self._check_circuit_breaker_cooldown():
                return False, f"Circuit breaker active: {self.circuit_breaker_reason}"

        order_limits = self.limits["order_limits"]

        # Check minimum order size
        if size < order_limits["min_order_size_base"]:
            return False, f"Order size {size} below minimum {order_limits['min_order_size_base']}"

        # Check maximum order size
        if size > order_limits["max_order_size_base"]:
            return False, f"Order size {size} exceeds maximum {order_limits['max_order_size_base']}"

        # Check order value
        order_value = size * price
        if order_value > order_limits["max_order_size_quote"]:
            return False, f"Order value {order_value} exceeds maximum {order_limits['max_order_size_quote']}"

        # Check order value vs book depth
        if current_book_depth is not None:
            max_pct = order_limits["max_order_value_pct_book"]
            if order_value > current_book_depth * max_pct:
                return False, f"Order value {order_value} exceeds {max_pct*100}% of book depth"

        # Check position limits after this order
        new_position = self.current_position_base
        if side == "buy":
            new_position += size
        else:
            new_position -= size

        position_valid, position_error = self._check_position_limits(
            new_position,
            order_value
        )
        if not position_valid:
            return False, position_error

        # Check rate limits
        rate_valid, rate_error = self._check_rate_limits()
        if not rate_valid:
            return False, rate_error

        return True, None

    def _check_position_limits(
        self,
        new_position_base: float,
        order_value: float
    ) -> Tuple[bool, Optional[str]]:
        """Check if new position would violate limits.

        Args:
            new_position_base: New position in base currency
            order_value: Order value in quote currency

        Returns:
            Tuple of (is_valid, error_message)
        """
        position_limits = self.limits["position_limits"]

        # Check absolute position limit
        max_pos = position_limits["max_position_base"]
        if abs(new_position_base) > max_pos:
            return False, f"Position {new_position_base} would exceed limit {max_pos}"

        # Check inventory skew
        max_skew = position_limits["max_inventory_skew"]
        if max_pos > 0:
            skew = abs(new_position_base) / max_pos
            if skew > max_skew:
                return False, f"Inventory skew {skew:.2%} would exceed limit {max_skew:.2%}"

        # Check quote currency position limit
        new_position_quote = self.current_position_quote + order_value
        max_quote = position_limits["max_position_quote"]
        if abs(new_position_quote) > max_quote:
            return False, f"Quote position {new_position_quote} would exceed limit {max_quote}"

        # Warning threshold check
        warning_threshold = position_limits.get("warning_threshold_pct", 0.7)
        if abs(new_position_base) > max_pos * warning_threshold:
            self.logger.warning(
                f"Position approaching limit: {new_position_base:.4f} / {max_pos} "
                f"({abs(new_position_base)/max_pos:.2%})"
            )

        return True, None

    def _check_rate_limits(self) -> Tuple[bool, Optional[str]]:
        """Check order rate limits.

        Returns:
            Tuple of (is_valid, error_message)
        """
        now = datetime.now()
        order_limits = self.limits["order_limits"]

        # Check orders per second
        one_second_ago = now - timedelta(seconds=1)
        recent_orders = sum(1 for ts in self.order_timestamps if ts > one_second_ago)

        max_per_second = order_limits["max_orders_per_second"]
        if recent_orders >= max_per_second:
            return False, f"Rate limit: {recent_orders} orders in last second (max {max_per_second})"

        # Check orders per minute
        one_minute_ago = now - timedelta(minutes=1)
        recent_orders_minute = sum(1 for ts in self.order_timestamps if ts > one_minute_ago)

        max_per_minute = order_limits["max_orders_per_minute"]
        if recent_orders_minute >= max_per_minute:
            return False, f"Rate limit: {recent_orders_minute} orders in last minute (max {max_per_minute})"

        return True, None

    def record_order(self) -> None:
        """Record order timestamp for rate limiting."""
        self.order_timestamps.append(datetime.now())

    def record_cancel(self) -> None:
        """Record cancel timestamp for rate limiting."""
        self.cancel_timestamps.append(datetime.now())

    def update_position(self, base_delta: float, quote_delta: float) -> None:
        """Update current position after trade.

        Args:
            base_delta: Change in base currency position
            quote_delta: Change in quote currency position
        """
        self.current_position_base += base_delta
        self.current_position_quote += quote_delta

        self.logger.info(
            f"Position updated: base={self.current_position_base:.4f}, "
            f"quote={self.current_position_quote:.2f}"
        )

    def record_trade_pnl(self, pnl: float) -> None:
        """Record trade PnL and check loss limits.

        Args:
            pnl: Trade profit/loss

        Raises:
            RiskViolation: If loss limits exceeded
            CircuitBreakerTripped: If circuit breaker triggered
        """
        loss_limits = self.limits["loss_limits"]

        # Update PnL tracking
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl

        # Check if loss
        if pnl < 0:
            self.trade_losses.append(abs(pnl))

            # Check per-trade loss limit
            if abs(pnl) > loss_limits["max_loss_per_trade"]:
                self._trigger_circuit_breaker(
                    f"Single trade loss {abs(pnl):.2f} exceeds limit {loss_limits['max_loss_per_trade']}"
                )
                raise CircuitBreakerTripped(self.circuit_breaker_reason)

            # Track consecutive losses
            if self.last_trade_profitable:
                self.consecutive_losses = 1
                self.last_trade_profitable = False
            else:
                self.consecutive_losses += 1

            # Check consecutive loss limit
            max_consecutive = self.limits["operational"]["max_consecutive_losses"]
            if self.consecutive_losses >= max_consecutive:
                self._trigger_circuit_breaker(
                    f"Consecutive losses ({self.consecutive_losses}) reached limit ({max_consecutive})"
                )
                raise CircuitBreakerTripped(self.circuit_breaker_reason)
        else:
            self.consecutive_losses = 0
            self.last_trade_profitable = True

        # Check daily loss limit
        if self.daily_pnl < -loss_limits["max_loss_per_day"]:
            if loss_limits.get("stop_trading_on_daily_loss", True):
                self._trigger_circuit_breaker(
                    f"Daily loss {abs(self.daily_pnl):.2f} exceeds limit {loss_limits['max_loss_per_day']}"
                )
                raise CircuitBreakerTripped(self.circuit_breaker_reason)

        # Check weekly loss limit
        if self.weekly_pnl < -loss_limits["max_loss_per_week"]:
            self._trigger_circuit_breaker(
                f"Weekly loss {abs(self.weekly_pnl):.2f} exceeds limit {loss_limits['max_loss_per_week']}"
            )
            raise CircuitBreakerTripped(self.circuit_breaker_reason)

        # Check monthly loss limit
        if self.monthly_pnl < -loss_limits["max_loss_per_month"]:
            self._trigger_circuit_breaker(
                f"Monthly loss {abs(self.monthly_pnl):.2f} exceeds limit {loss_limits['max_loss_per_month']}"
            )
            raise CircuitBreakerTripped(self.circuit_breaker_reason)

        self.logger.info(
            f"PnL recorded: trade={pnl:.2f}, daily={self.daily_pnl:.2f}, "
            f"weekly={self.weekly_pnl:.2f}, monthly={self.monthly_pnl:.2f}"
        )

    def validate_market_conditions(
        self,
        spread_bps: float,
        book_depth: float,
        price_change_pct: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate current market conditions against limits.

        Args:
            spread_bps: Current bid-ask spread in basis points
            book_depth: Order book depth in quote currency
            price_change_pct: Recent price change percentage

        Returns:
            Tuple of (is_valid, error_message)
        """
        market_limits = self.limits["market_conditions"]

        # Check spread
        if spread_bps > market_limits["max_spread_bps"]:
            return False, f"Spread {spread_bps:.1f} bps exceeds limit {market_limits['max_spread_bps']}"

        # Check book depth
        if book_depth < market_limits["min_book_depth_quote"]:
            return False, f"Book depth {book_depth:.2f} below minimum {market_limits['min_book_depth_quote']}"

        # Check price volatility
        if price_change_pct is not None:
            max_move = market_limits["max_price_move_per_second_pct"]
            if abs(price_change_pct) > max_move:
                return False, f"Price move {abs(price_change_pct):.2%} exceeds limit {max_move:.2%}"

        return True, None

    def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger circuit breaker.

        Args:
            reason: Reason for circuit breaker activation
        """
        if not self.limits["circuit_breakers"]["enable_circuit_breakers"]:
            return

        self.circuit_breaker_active = True
        self.circuit_breaker_time = datetime.now()
        self.circuit_breaker_reason = reason

        self.logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")

    def _check_circuit_breaker_cooldown(self) -> bool:
        """Check if circuit breaker cooldown has elapsed.

        Returns:
            True if cooldown elapsed and breaker can be reset
        """
        if not self.circuit_breaker_active or self.circuit_breaker_time is None:
            return True

        cb_config = self.limits["circuit_breakers"]
        cooldown = timedelta(seconds=cb_config["cooldown_period_seconds"])

        if datetime.now() - self.circuit_breaker_time > cooldown:
            if cb_config["auto_resume_after_cooldown"]:
                self.reset_circuit_breaker()
                return True
            else:
                self.logger.warning(
                    "Circuit breaker cooldown elapsed but manual override required"
                )

        return False

    def reset_circuit_breaker(self, manual_override: bool = False) -> None:
        """Reset circuit breaker after cooldown or manual override.

        Args:
            manual_override: Whether this is a manual override

        Raises:
            RiskViolation: If manual override required but not provided
        """
        cb_config = self.limits["circuit_breakers"]

        if cb_config["require_manual_override"] and not manual_override:
            raise RiskViolation(
                "Circuit breaker requires manual override to reset"
            )

        self.circuit_breaker_active = False
        self.circuit_breaker_time = None
        self.circuit_breaker_reason = None

        self.logger.info("Circuit breaker reset")

    def reset_daily_limits(self) -> None:
        """Reset daily tracking (call at start of trading day)."""
        self.daily_pnl = 0.0
        self.daily_loss_start = datetime.now()
        self.consecutive_losses = 0
        self.last_trade_profitable = True
        self.logger.info("Daily limits reset")

    def reset_weekly_limits(self) -> None:
        """Reset weekly tracking."""
        self.weekly_pnl = 0.0
        self.weekly_loss_start = datetime.now()
        self.logger.info("Weekly limits reset")

    def reset_monthly_limits(self) -> None:
        """Reset monthly tracking."""
        self.monthly_pnl = 0.0
        self.monthly_loss_start = datetime.now()
        self.logger.info("Monthly limits reset")

    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status summary.

        Returns:
            Dictionary with current risk metrics
        """
        position_limits = self.limits["position_limits"]
        loss_limits = self.limits["loss_limits"]

        return {
            "circuit_breaker_active": self.circuit_breaker_active,
            "circuit_breaker_reason": self.circuit_breaker_reason,
            "position": {
                "base": self.current_position_base,
                "quote": self.current_position_quote,
                "utilization_pct": abs(self.current_position_base) / position_limits["max_position_base"] * 100
                if position_limits["max_position_base"] > 0 else 0,
            },
            "pnl": {
                "daily": self.daily_pnl,
                "weekly": self.weekly_pnl,
                "monthly": self.monthly_pnl,
                "daily_limit_remaining": loss_limits["max_loss_per_day"] + self.daily_pnl,
            },
            "consecutive_losses": self.consecutive_losses,
            "rate_limits": {
                "orders_last_minute": sum(
                    1 for ts in self.order_timestamps
                    if ts > datetime.now() - timedelta(minutes=1)
                ),
                "cancels_last_minute": sum(
                    1 for ts in self.cancel_timestamps
                    if ts > datetime.now() - timedelta(minutes=1)
                ),
            }
        }
