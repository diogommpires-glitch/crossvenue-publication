"""Limit order book with price-time priority matching.

This module implements a high-performance limit order book using sorted containers
for efficient price-level management and FIFO queues for time priority.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sortedcontainers import SortedDict

from .order import Order, Fill
from ..utils.types import Side, OrderType, OrderStatus, Price, Quantity, Timestamp


@dataclass
class BookLevel:
    """A single price level in the order book.

    Maintains FIFO queue of orders at this price level for time priority.
    """

    price: Price
    orders: deque = None  # deque of Order objects
    total_quantity: Quantity = Quantity(0)

    def __post_init__(self) -> None:
        if self.orders is None:
            self.orders = deque()

    def add_order(self, order: Order) -> None:
        """Add order to the back of the queue (time priority)."""
        self.orders.append(order)
        self.total_quantity = Quantity(self.total_quantity + order.quantity)

    def remove_order(self, order_id: str) -> Optional[Order]:
        """Remove specific order by ID. O(n) operation."""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                removed = self.orders[i]
                del self.orders[i]
                self.total_quantity = Quantity(self.total_quantity - removed.quantity)
                return removed
        return None

    def is_empty(self) -> bool:
        """Check if level has no orders."""
        return len(self.orders) == 0


class LimitOrderBook:
    """Limit order book with price-time priority matching.

    Architecture:
    - Bids: SortedDict with descending price (best bid = highest price)
    - Asks: SortedDict with ascending price (best ask = lowest price)
    - Each price level maintains FIFO queue for time priority
    - O(log n) insertion and cancellation
    - O(1) best bid/ask access

    Critical implementation details:
    - Price-time priority strictly enforced
    - Self-matching prevention (same trader_id)
    - Atomic fill generation (both sides get Fill records)
    - Proper order status tracking
    """

    def __init__(self) -> None:
        # Bids: max-heap (negative keys for descending order in SortedDict)
        self._bids: SortedDict[float, BookLevel] = SortedDict()

        # Asks: min-heap (natural ascending order)
        self._asks: SortedDict[float, BookLevel] = SortedDict()

        # Track active orders by ID for O(1) lookup
        self._orders: Dict[str, Order] = {}

        # Track which price level each order is at (for fast cancellation)
        self._order_to_level: Dict[str, Tuple[Side, Price]] = {}

        # Statistics
        self.total_fills: int = 0
        self.total_volume: int = 0

    def add_order(self, order: Order) -> Tuple[Order, List[Fill]]:
        """Add order to the book and attempt to match.

        Returns:
            Tuple of (updated_order, list_of_fills)
            - If fully filled, order status is FILLED
            - If partially filled, order status is PARTIALLY_FILLED
            - If no fills, order status is ACTIVE
        """
        # Validate order
        if order.order_id in self._orders:
            return order.with_status(OrderStatus.REJECTED), []

        # Market orders must match immediately or reject
        if order.order_type == OrderType.MARKET:
            return self._match_market_order(order)

        # Limit order: try to match aggressively
        fills, remaining_qty = self._match_limit_order(order)

        # Determine final status
        if remaining_qty == 0:
            final_order = order.with_status(OrderStatus.FILLED)
        elif remaining_qty < order.quantity:
            # Partially filled - add remainder to book
            remainder = Order(
                order_id=order.order_id,
                side=order.side,
                order_type=order.order_type,
                quantity=Quantity(remaining_qty),
                price=order.price,
                timestamp=order.timestamp,
                trader_id=order.trader_id,
                status=OrderStatus.PARTIALLY_FILLED,
            )
            self._add_to_book(remainder)
            final_order = remainder
        else:
            # Not filled at all - add to book
            active_order = order.with_status(OrderStatus.ACTIVE)
            self._add_to_book(active_order)
            final_order = active_order

        return final_order, fills

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order.

        Returns:
            True if order was cancelled, False if not found
        """
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        side, price = self._order_to_level[order_id]

        # Remove from appropriate side
        levels = self._bids if side == Side.BUY else self._asks
        price_key = -float(price) if side == Side.BUY else float(price)

        if price_key in levels:
            level = levels[price_key]
            level.remove_order(order_id)

            # Clean up empty levels
            if level.is_empty():
                del levels[price_key]

        # Clean up tracking
        del self._orders[order_id]
        del self._order_to_level[order_id]

        return True

    def get_best_bid(self) -> Optional[Price]:
        """Get best bid price (highest buy price)."""
        if not self._bids:
            return None
        # First key in sorted dict (most negative = highest actual price)
        return Price(-self._bids.keys()[0])

    def get_best_ask(self) -> Optional[Price]:
        """Get best ask price (lowest sell price)."""
        if not self._asks:
            return None
        return Price(self._asks.keys()[0])

    def get_mid_price(self) -> Optional[Price]:
        """Get mid-price (average of best bid and ask)."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()

        if bid is None or ask is None:
            return None

        return Price((bid + ask) / 2.0)

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()

        if bid is None or ask is None:
            return None

        return float(ask - bid)

    def get_depth(self, side: Side, levels: int = 5) -> List[Tuple[Price, Quantity]]:
        """Get market depth (price, quantity) for top N levels."""
        book = self._bids if side == Side.BUY else self._asks
        result = []

        for i, (price_key, level) in enumerate(book.items()):
            if i >= levels:
                break
            actual_price = Price(-price_key if side == Side.BUY else price_key)
            result.append((actual_price, level.total_quantity))

        return result

    def _add_to_book(self, order: Order) -> None:
        """Add order to the appropriate book side."""
        assert order.price is not None, "Cannot add market order to book"

        # Select side and convert price key
        if order.side == Side.BUY:
            levels = self._bids
            price_key = -float(order.price)  # Negative for descending order
        else:
            levels = self._asks
            price_key = float(order.price)

        # Get or create level
        if price_key not in levels:
            levels[price_key] = BookLevel(price=order.price)

        # Add order to level
        levels[price_key].add_order(order)

        # Track order
        self._orders[order.order_id] = order
        self._order_to_level[order.order_id] = (order.side, order.price)

    def _match_limit_order(self, order: Order) -> Tuple[List[Fill], int]:
        """Match a limit order against the book.

        Returns:
            (list_of_fills, remaining_quantity)
        """
        fills: List[Fill] = []
        remaining = order.quantity

        # Determine which side to match against
        if order.side == Side.BUY:
            # Match against asks (sellers)
            levels = self._asks
            can_match = lambda ask_price: float(ask_price) <= float(order.price)
        else:
            # Match against bids (buyers)
            levels = self._bids
            can_match = lambda bid_price: float(-bid_price) >= float(order.price)

        # Match against best prices until no more matches or fully filled
        while remaining > 0 and levels:
            best_price_key = levels.keys()[0]
            level = levels[best_price_key]

            # Check if we can match at this price
            if not can_match(best_price_key):
                break

            # Match against orders at this level (FIFO)
            while remaining > 0 and not level.is_empty():
                resting_order = level.orders[0]

                # Prevent self-matching
                if resting_order.trader_id == order.trader_id:
                    # Skip this order and clean up tracking (fix memory leak)
                    level.orders.popleft()
                    level.total_quantity = Quantity(level.total_quantity - resting_order.quantity)
                    del self._orders[resting_order.order_id]
                    del self._order_to_level[resting_order.order_id]
                    continue

                # Determine fill quantity
                fill_qty = min(remaining, resting_order.quantity)
                actual_price = Price(float(resting_order.price))

                # Create fills for both sides
                # Aggressor fill (incoming order)
                aggressor_fill = Fill(
                    order_id=order.order_id,
                    side=order.side,
                    quantity=Quantity(fill_qty),
                    price=actual_price,
                    timestamp=order.timestamp,
                    trader_id=order.trader_id,
                    is_aggressor=True,
                )
                fills.append(aggressor_fill)

                # Passive fill (resting order)
                passive_fill = Fill(
                    order_id=resting_order.order_id,
                    side=resting_order.side,
                    quantity=Quantity(fill_qty),
                    price=actual_price,
                    timestamp=order.timestamp,
                    trader_id=resting_order.trader_id,
                    is_aggressor=False,
                )
                fills.append(passive_fill)

                # Update quantities
                remaining -= fill_qty
                self.total_fills += 2  # Both sides
                self.total_volume += fill_qty

                # Handle resting order
                if fill_qty == resting_order.quantity:
                    # Fully filled - remove from book
                    level.orders.popleft()
                    level.total_quantity = Quantity(level.total_quantity - fill_qty)
                    del self._orders[resting_order.order_id]
                    del self._order_to_level[resting_order.order_id]
                else:
                    # Partially filled - update quantity
                    updated = Order(
                        order_id=resting_order.order_id,
                        side=resting_order.side,
                        order_type=resting_order.order_type,
                        quantity=Quantity(resting_order.quantity - fill_qty),
                        price=resting_order.price,
                        timestamp=resting_order.timestamp,
                        trader_id=resting_order.trader_id,
                        status=OrderStatus.PARTIALLY_FILLED,
                    )
                    level.orders[0] = updated
                    level.total_quantity = Quantity(level.total_quantity - fill_qty)
                    self._orders[resting_order.order_id] = updated

            # Clean up empty level
            if level.is_empty():
                del levels[best_price_key]

        return fills, remaining

    def _match_market_order(self, order: Order) -> Tuple[Order, List[Fill]]:
        """Match a market order atomically (must fill completely or reject without mutating book).

        v0.3.0: Fixed atomicity - simulate matching first, only apply if fully fillable.
        """
        # Phase 1: Simulate matching to check if order can be fully filled
        # Track what we would consume without mutating the book
        remaining = order.quantity
        simulated_fills: List[Tuple[Order, int]] = []  # (resting_order, fill_qty)

        levels = self._asks if order.side == Side.BUY else self._bids

        # Track self-matched orders to clean up later
        self_matched_orders: List[Tuple[str, Side, Price]] = []

        # Walk through levels to simulate fills
        for price_key in list(levels.keys()):
            level = levels[price_key]

            for resting_order in level.orders:
                if remaining <= 0:
                    break

                # Skip self-matching (track for cleanup)
                if resting_order.trader_id == order.trader_id:
                    self_matched_orders.append((resting_order.order_id, resting_order.side, resting_order.price))
                    continue

                # Determine fill quantity
                fill_qty = min(remaining, resting_order.quantity)
                simulated_fills.append((resting_order, fill_qty))
                remaining -= fill_qty

            if remaining <= 0:
                break

        # Phase 2: Check if order can be fully filled
        if remaining > 0:
            # Insufficient liquidity - reject without mutating book
            # BUT: clean up self-matched orders (BUG FIX: memory leak prevention)
            for order_id, side, price in self_matched_orders:
                price_key = -float(price) if side == Side.BUY else float(price)
                if price_key in levels:
                    level = levels[price_key]
                    level.remove_order(order_id)
                    if level.is_empty():
                        del levels[price_key]
                # Clean up tracking
                if order_id in self._orders:
                    del self._orders[order_id]
                if order_id in self._order_to_level:
                    del self._order_to_level[order_id]

            return order.with_status(OrderStatus.REJECTED), []

        # Phase 3: Clean up self-matched orders (BUG FIX: memory leak prevention)
        for order_id, side, price in self_matched_orders:
            price_key = -float(price) if side == Side.BUY else float(price)
            if price_key in levels:
                level = levels[price_key]
                level.remove_order(order_id)
                if level.is_empty():
                    del levels[price_key]
            # Clean up tracking
            if order_id in self._orders:
                del self._orders[order_id]
            if order_id in self._order_to_level:
                del self._order_to_level[order_id]

        # Phase 4: Order can be fully filled - now apply mutations and generate fills
        fills: List[Fill] = []

        for resting_order, fill_qty in simulated_fills:
            actual_price = Price(float(resting_order.price))

            # Create fills for both sides
            aggressor_fill = Fill(
                order_id=order.order_id,
                side=order.side,
                quantity=Quantity(fill_qty),
                price=actual_price,
                timestamp=order.timestamp,
                trader_id=order.trader_id,
                is_aggressor=True,
            )
            fills.append(aggressor_fill)

            passive_fill = Fill(
                order_id=resting_order.order_id,
                side=resting_order.side,
                quantity=Quantity(fill_qty),
                price=actual_price,
                timestamp=order.timestamp,
                trader_id=resting_order.trader_id,
                is_aggressor=False,
            )
            fills.append(passive_fill)

            # Update statistics
            self.total_fills += 2
            self.total_volume += fill_qty

            # Mutate book: handle resting order
            side = resting_order.side
            price = resting_order.price
            price_key = -float(price) if side == Side.BUY else float(price)
            level = levels[price_key]

            if fill_qty == resting_order.quantity:
                # Fully filled - remove from book
                level.orders.popleft()
                level.total_quantity = Quantity(level.total_quantity - fill_qty)
                del self._orders[resting_order.order_id]
                del self._order_to_level[resting_order.order_id]
            else:
                # Partially filled - update quantity
                updated = Order(
                    order_id=resting_order.order_id,
                    side=resting_order.side,
                    order_type=resting_order.order_type,
                    quantity=Quantity(resting_order.quantity - fill_qty),
                    price=resting_order.price,
                    timestamp=resting_order.timestamp,
                    trader_id=resting_order.trader_id,
                    status=OrderStatus.PARTIALLY_FILLED,
                )
                level.orders[0] = updated
                level.total_quantity = Quantity(level.total_quantity - fill_qty)
                self._orders[resting_order.order_id] = updated

            # Clean up empty level
            if level.is_empty():
                del levels[price_key]

        return order.with_status(OrderStatus.FILLED), fills

    def __str__(self) -> str:
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        mid = self.get_mid_price()
        spread = self.get_spread()

        return (
            f"LOB(bid={bid}, ask={ask}, mid={mid}, spread={spread}, "
            f"fills={self.total_fills}, volume={self.total_volume})"
        )
