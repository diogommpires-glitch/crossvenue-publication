"""Avellaneda-Stoikov optimal market-making strategy.

Implementation of the HJB-based optimal market-making framework.

Reference:
- Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
- Extends baseline with inventory-aware pricing and risk aversion

Key features:
- Reservation price: adjusts quotes based on inventory position
- Optimal spread: balances fill probability vs adverse selection
- Risk aversion parameter: controls inventory risk tolerance
- Time horizon: incorporates finite trading window

The strategy computes:
1. Reservation price: r = s - q·γ·σ²·(T-t)
   where q = inventory, γ = risk aversion, σ = volatility, T-t = time remaining

2. Optimal spread: δ = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/κ)
   where κ = order arrival intensity

In practice, we use simplified versions calibrated to market data.
"""

from typing import Dict, List
import numpy as np

from .base import Strategy
from ..orderbook.order import Order
from ..utils.types import Side, OrderType, Price, Quantity, Timestamp


class AvellanedaStoikov(Strategy):
    """Avellaneda-Stoikov optimal market maker with inventory risk management.

    This strategy implements the HJB framework for optimal market making,
    adjusting quotes based on:
    - Current inventory position
    - Estimated market volatility
    - Risk aversion parameter
    - Time remaining in trading horizon

    Unlike the naive strategy, this adapts spreads and skews quotes to
    manage inventory risk and maximize expected terminal wealth.

    Parameters:
    ----------
    risk_aversion : float
        Risk aversion parameter γ (gamma). Higher = more conservative.
        Typical values: 0.01 - 0.1

    volatility : float
        Estimated price volatility σ (annual)

    order_size : float
        Size of each quote in BASE ASSET units (BTC for BTC/USDT pair).
        Example: 0.01 BTC = ~$670 at $67k BTC price

    max_inventory : float
        Maximum absolute inventory position in BASE ASSET units (BTC).
        Example: 0.1 BTC = ~$6,700 at $67k BTC price

    time_horizon : float
        Trading time horizon T (seconds)

    order_intensity : float
        Expected order arrival rate κ (kappa)
    """

    def __init__(
        self,
        risk_aversion: float = 0.05,
        volatility: float = 0.20,
        order_size: float = 0.01,
        max_inventory: float = 0.1,
        time_horizon: float = 3600.0,  # 1 hour
        order_intensity: float = 1.0,
        strategy_id: str = "as_mm",
        seconds_per_year: float = 365.25 * 24 * 3600,
        maker_fee: float = 0.001,
        taker_fee: float = 0.001,
    ) -> None:
        super().__init__(strategy_id, maker_fee, taker_fee)
        if not np.isfinite(seconds_per_year) or seconds_per_year <= 0:
            raise ValueError("seconds_per_year must be finite and positive")
        self.seconds_per_year = float(seconds_per_year)
        self.risk_aversion = risk_aversion  # γ (gamma)
        self.volatility = volatility  # σ (sigma)
        self.order_size = order_size
        self.max_inventory = max_inventory
        self.time_horizon = time_horizon  # T
        self.order_intensity = order_intensity  # κ (kappa)

        # Convert annual volatility to per-second
        # σ_annual = σ_second * sqrt(seconds_per_year)
        self.vol_per_second = self.volatility / np.sqrt(self.seconds_per_year)

        # Track strategy start time
        self.start_time: float = 0.0
        self.initialized: bool = False

        # Track last quote time
        self.last_quote_time: float = 0.0
        self.quote_interval: float = 1.0

    def on_market_event(self, market_state: Dict) -> List[Order]:
        """Generate optimal quotes using Avellaneda-Stoikov framework."""
        current_time = market_state["time"]
        mid_price = market_state["mid_price"]

        # Initialize start time on first call
        if not self.initialized:
            self.start_time = current_time
            self.initialized = True

        # Rate limit quoting
        if current_time - self.last_quote_time < self.quote_interval:
            return []

        self.last_quote_time = current_time

        # Calculate time remaining
        time_remaining = max(0.0, self.time_horizon - (current_time - self.start_time))

        # If time expired, start winding down (widen spreads)
        if time_remaining < 60.0:  # Last minute
            # Aggressive unwinding
            return self._generate_unwind_orders(mid_price, current_time)

        # Calculate reservation price
        reservation_price = self._calculate_reservation_price(
            mid_price, self.inventory, time_remaining
        )

        # Calculate optimal spread
        optimal_spread = self._calculate_optimal_spread(time_remaining)

        # Check inventory limits (order_size is in base asset units)
        can_buy = abs(self.inventory + self.order_size) <= self.max_inventory
        can_sell = abs(self.inventory - self.order_size) <= self.max_inventory

        orders = []

        # Place buy quote (bid)
        if can_buy:
            # Bid is below reservation price
            bid_price = reservation_price - optimal_spread / 2.0
            bid = Order(
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                quantity=Quantity(self.order_size),
                price=Price(max(0.01, bid_price)),
                timestamp=Timestamp(current_time),
                trader_id=self.strategy_id,
            )
            orders.append(bid)

        # Place sell quote (ask)
        if can_sell:
            # Ask is above reservation price
            ask_price = reservation_price + optimal_spread / 2.0
            ask = Order(
                side=Side.SELL,
                order_type=OrderType.LIMIT,
                quantity=Quantity(self.order_size),
                price=Price(ask_price),
                timestamp=Timestamp(current_time),
                trader_id=self.strategy_id,
            )
            orders.append(ask)

        return orders

    def _calculate_reservation_price(
        self,
        mid_price: float,
        inventory: int,
        time_remaining: float
    ) -> float:
        """Calculate reservation price: r = s - q·γ·σ²·(T-t)

        The reservation price is the "indifference price" at which the
        market maker is indifferent between buying and selling.

        - Positive inventory (long) → reservation < mid (incentive to sell)
        - Negative inventory (short) → reservation > mid (incentive to buy)
        """
        # Inventory penalty term
        inventory_penalty = (
            inventory *
            self.risk_aversion *
            (self.vol_per_second ** 2) *
            time_remaining
        )

        reservation = mid_price - inventory_penalty

        return reservation

    def _calculate_optimal_spread(self, time_remaining: float) -> float:
        """Calculate optimal spread: δ = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/κ)

        The optimal spread balances:
        - Wider spreads → lower fill rate but better prices
        - Narrower spreads → higher fill rate but worse prices

        The formula has two components:
        1. Risk component: γ·σ²·(T-t) (increases with risk aversion and time)
        2. Fill probability component: (2/γ)·ln(1 + γ/κ) (based on order flow)
        """
        # Risk component
        risk_component = (
            self.risk_aversion *
            (self.vol_per_second ** 2) *
            time_remaining
        )

        # Fill probability component
        # Uses order arrival intensity κ
        fill_component = (
            (2.0 / self.risk_aversion) *
            np.log(1.0 + self.risk_aversion / self.order_intensity)
        )

        optimal_spread = risk_component + fill_component

        # Clamp to reasonable range
        optimal_spread = np.clip(optimal_spread, 0.02, 1.0)

        return optimal_spread

    def _generate_unwind_orders(self, mid_price: float, current_time: float) -> List[Order]:
        """Generate aggressive orders to unwind position near end of horizon.

        When approaching the end of the trading horizon, the strategy
        needs to close out inventory to avoid carrying risk.

        Chunks large positions into max_order_size pieces to respect risk limits.
        """
        if self.inventory == 0:
            return []

        orders = []

        # Determine unwind direction
        if self.inventory > 0:
            # Long inventory → need to sell
            # Respect max_order_size by capping unwind size
            unwind_size = min(self.order_size, abs(self.inventory))
            ask = Order(
                side=Side.SELL,
                order_type=OrderType.LIMIT,
                quantity=Quantity(unwind_size),
                price=Price(mid_price - 0.01),  # Aggressive (inside spread)
                timestamp=Timestamp(current_time),
                trader_id=self.strategy_id,
            )
            orders.append(ask)

        elif self.inventory < 0:
            # Short inventory → need to buy
            # Respect max_order_size by capping unwind size
            unwind_size = min(self.order_size, abs(self.inventory))
            bid = Order(
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                quantity=Quantity(unwind_size),
                price=Price(mid_price + 0.01),  # Aggressive (inside spread)
                timestamp=Timestamp(current_time),
                trader_id=self.strategy_id,
            )
            orders.append(bid)

        return orders

    def get_diagnostics(self, mid_price: float, current_time: float) -> Dict:
        """Get strategy diagnostics for analysis."""
        time_remaining = max(0.0, self.time_horizon - (current_time - self.start_time))

        reservation = self._calculate_reservation_price(
            mid_price, self.inventory, time_remaining
        )

        optimal_spread = self._calculate_optimal_spread(time_remaining)

        return {
            "inventory": self.inventory,
            "cash": self.cash,
            "pnl": self.get_unrealized_pnl(mid_price),
            "reservation_price": reservation,
            "optimal_spread": optimal_spread,
            "inventory_skew": reservation - mid_price,
            "time_remaining": time_remaining,
            "total_fills": len(self.fills),
        }

    def __str__(self) -> str:
        return (
            f"AvellanedaStoikov(gamma={self.risk_aversion:.3f}, "
            f"sigma={self.volatility:.2f}, inv={self.inventory})"
        )
