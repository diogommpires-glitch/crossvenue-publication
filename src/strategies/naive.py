"""Naive market-making strategy (baseline for comparison).

This is the baseline strategy that does NOT account for self-impact.
It serves as a comparison point to demonstrate the importance of self-impact modeling.

Key characteristics:
- Fixed spread around mid-price
- Simple inventory limits
- No self-impact adjustment
- Expected to overstate profitability vs self-impact-aware version
"""

from typing import Dict, List
import numpy as np

from .base import Strategy
from ..orderbook.order import Order
from ..utils.types import Side, OrderType, Price, Quantity, Timestamp


class NaiveMarketMaker(Strategy):
    """Naive market maker with fixed spreads and no self-impact awareness.

    This strategy:
    1. Quotes bid/ask at fixed spread around mid-price
    2. Enforces simple inventory limits
    3. Uses mid-price directly (ignores self-impact)

    Parameters:
    ----------
    half_spread : float
        Half of the bid-ask spread to quote
    order_size : int
        Size of each quote in BASE ASSET units (e.g., BTC for BTC/USDT pair)
    max_inventory : int
        Maximum absolute inventory position
    """

    def __init__(
        self,
        half_spread: float = 0.05,
        order_size: int = 100,
        max_inventory: int = 500,
        strategy_id: str = "naive_mm",
    ) -> None:
        super().__init__(strategy_id)
        self.half_spread = half_spread
        self.order_size = order_size
        self.max_inventory = max_inventory

        # Track last quote time to avoid over-quoting
        self.last_quote_time: float = 0.0
        self.quote_interval: float = 1.0  # Quote every N seconds

    def on_market_event(self, market_state: Dict) -> List[Order]:
        """Generate naive quotes around mid-price.

        Simply places bid and ask at fixed spread, subject to inventory limits.
        """
        current_time = market_state["time"]
        mid_price = market_state["mid_price"]

        # Rate limit quoting
        if current_time - self.last_quote_time < self.quote_interval:
            return []

        self.last_quote_time = current_time

        # Check if we're at inventory limits
        can_buy = abs(self.inventory + self.order_size) <= self.max_inventory
        can_sell = abs(self.inventory - self.order_size) <= self.max_inventory

        orders = []

        # Place buy quote
        if can_buy:
            bid_price = mid_price - self.half_spread
            bid = Order(
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                quantity=Quantity(self.order_size),
                price=Price(max(0.01, bid_price)),  # Ensure positive
                timestamp=Timestamp(current_time),
                trader_id=self.strategy_id,
            )
            orders.append(bid)

        # Place sell quote
        if can_sell:
            ask_price = mid_price + self.half_spread
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

    def __str__(self) -> str:
        return (
            f"NaiveMarketMaker(spread={self.half_spread*2:.3f}, "
            f"size={self.order_size}, inv={self.inventory})"
        )
