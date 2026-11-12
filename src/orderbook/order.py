"""Order data structures.

This module defines Order and Fill classes representing the fundamental
units of trading activity in the limit order book.
"""

from dataclasses import dataclass, field
from typing import Optional
import uuid
import math

from ..utils.types import Side, OrderType, OrderStatus, Price, Quantity, Timestamp


@dataclass(frozen=True)
class Order:
    """Immutable order representation.

    Orders are immutable to prevent accidental modification after submission.
    Any changes (fills, cancellations) create new Order instances or Fill records.

    Attributes:
        order_id: Unique identifier (auto-generated if not provided)
        side: BUY or SELL
        order_type: LIMIT or MARKET
        quantity: Number of shares/contracts
        price: Limit price (None for market orders)
        timestamp: Submission timestamp
        trader_id: Identifier of the submitting trader/strategy
        status: Current order status
    """

    side: Side
    order_type: OrderType
    quantity: Quantity
    timestamp: Timestamp
    trader_id: str
    price: Optional[Price] = None
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: OrderStatus = OrderStatus.PENDING
    # Optional for legacy single-instrument callers. Cross-asset orders must
    # carry the canonical instrument ID so risk and accounting cannot mix units.
    instrument_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate order parameters."""
        # Validate quantity
        if not math.isfinite(float(self.quantity)) or self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")

        # Validate price for limit orders
        if self.order_type == OrderType.LIMIT:
            if self.price is None:
                raise ValueError("Limit orders must have a price")
            if not math.isfinite(float(self.price)) or (self.price <= 0 and self.instrument_id is None):
                raise ValueError(f"Order price must be positive for legacy instruments, got {self.price}")

        # Market orders should not have a price
        if self.order_type == OrderType.MARKET and self.price is not None:
            raise ValueError("Market orders cannot have a price")

    def with_status(self, status: OrderStatus) -> "Order":
        """Create a copy with updated status."""
        return Order(
            order_id=self.order_id,
            side=self.side,
            order_type=self.order_type,
            quantity=self.quantity,
            price=self.price,
            timestamp=self.timestamp,
            trader_id=self.trader_id,
            status=status,
            instrument_id=self.instrument_id,
        )

    def __str__(self) -> str:
        price_str = f"@{self.price}" if self.price else "MKT"
        return f"Order({self.order_id[:8]} {self.side} {self.quantity} {price_str})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class Fill:
    """Record of an executed trade.

    A Fill represents a match between a buyer and seller. Each side of the
    trade receives a Fill record with their perspective.

    Attributes:
        fill_id: Unique identifier for this fill
        order_id: Order that was filled
        side: BUY or SELL (from the filled order's perspective)
        quantity: Number of shares/contracts filled
        price: Execution price
        timestamp: Execution timestamp
        trader_id: Identifier of the trader whose order was filled
        is_aggressor: True if this order crossed the spread (took liquidity)
    """

    order_id: str
    side: Side
    quantity: Quantity
    price: Price
    timestamp: Timestamp
    trader_id: str
    is_aggressor: bool
    fill_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # `estimated` distinguishes modeled paper fills from observed executions.
    # FX quote-based fills set this explicitly and retain their triggering event.
    instrument_id: Optional[str] = None
    source_event_id: Optional[str] = None
    execution_model: str = "legacy"
    estimated: bool = False

    def __post_init__(self) -> None:
        """Validate fill parameters."""
        if not math.isfinite(float(self.quantity)) or self.quantity <= 0:
            raise ValueError(f"Fill quantity must be positive, got {self.quantity}")
        if not math.isfinite(float(self.price)) or (self.price <= 0 and self.instrument_id is None):
            raise ValueError(f"Fill price must be positive for legacy instruments, got {self.price}")

    def __str__(self) -> str:
        aggressor_flag = "AGG" if self.is_aggressor else "PAS"
        return (
            f"Fill({self.fill_id[:8]} {self.side} {self.quantity}@{self.price} "
            f"{aggressor_flag})"
        )

    def __repr__(self) -> str:
        return self.__str__()
