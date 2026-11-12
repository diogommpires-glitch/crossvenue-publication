"""Core type definitions for the market simulator.

This module defines the fundamental types used throughout the simulator,
ensuring type safety and clear semantics.
"""

from enum import Enum, auto
from typing import NewType

# Type aliases for clarity and type safety
Price = NewType("Price", float)
Quantity = NewType("Quantity", float)
Timestamp = NewType("Timestamp", float)


class Side(Enum):
    """Order side: BUY or SELL."""

    BUY = auto()
    SELL = auto()

    def opposite(self) -> "Side":
        """Return the opposite side."""
        return Side.SELL if self == Side.BUY else Side.BUY

    def __str__(self) -> str:
        return "BUY" if self == Side.BUY else "SELL"


class OrderType(Enum):
    """Order type: LIMIT or MARKET."""

    LIMIT = auto()
    MARKET = auto()

    def __str__(self) -> str:
        return "LIMIT" if self == OrderType.LIMIT else "MARKET"


class OrderStatus(Enum):
    """Order lifecycle status."""

    PENDING = auto()      # Order submitted but not yet in book
    ACTIVE = auto()       # Order is active in the book
    FILLED = auto()       # Order completely filled
    PARTIALLY_FILLED = auto()  # Order partially filled
    CANCELLED = auto()    # Order cancelled
    REJECTED = auto()     # Order rejected (invalid parameters)

    def __str__(self) -> str:
        return self.name
