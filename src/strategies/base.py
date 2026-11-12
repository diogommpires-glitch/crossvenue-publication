"""Base strategy interface.

All trading strategies inherit from this ABC and implement the core methods.
"""

from abc import ABC, abstractmethod
import math
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..orderbook.order import Order, Fill
from ..utils.types import Side, Timestamp


@dataclass
class StrategyState:
    """Current state of a trading strategy."""

    time: Timestamp
    inventory: float  # Net position (positive = long, negative = short)
    cash: float  # Cash position
    unrealized_pnl: float  # Mark-to-market P&L
    realized_pnl: float  # Realized P&L from fills
    total_fills: int
    active_orders: List[str]  # Active order IDs


class Strategy(ABC):
    """Abstract base class for trading strategies.

    A strategy receives market state updates and decides when to place/cancel orders.
    It maintains internal state (inventory, P&L, risk limits) and implements
    a trading logic via the on_market_event() method.

    Convention: All order quantities are specified in BASE ASSET units.
    For a BTC/USDT pair, order_size=100 means 100 BTC, not 100 USDT.
    """

    def __init__(
        self,
        strategy_id: str = "strategy",
        maker_fee: float = 0.001,  # 0.1% maker fee
        taker_fee: float = 0.001,  # 0.1% taker fee
    ) -> None:
        if not math.isfinite(maker_fee) or not 0.0 <= maker_fee <= 0.1:
            raise ValueError("maker_fee must be finite and between 0 and 0.1")
        if not math.isfinite(taker_fee) or not 0.0 <= taker_fee <= 0.1:
            raise ValueError("taker_fee must be finite and between 0 and 0.1")
        self.strategy_id = strategy_id
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        # State
        self.inventory: float = 0.0
        self.cash: float = 0.0
        self.realized_pnl: float = 0.0  # Only from closed positions + fees
        self.cost_basis: float = 0.0  # Absolute entry notional for open position
        self.fees_paid: float = 0.0
        self.fills: List[Fill] = []
        self.active_order_ids: List[str] = []

    @abstractmethod
    def on_market_event(self, market_state: Dict) -> List[Order]:
        """Called on each market event to generate trading decisions.

        Parameters:
        ----------
        market_state : Dict
            Current market state from simulator

        Returns:
        -------
        List[Order]
            Orders to submit (empty list if no action)
        """
        pass

    def on_fill(self, fill: Fill, current_price: float) -> None:
        """Apply a fill using average-cost accounting, including its fee."""
        quantity = float(fill.quantity)
        price = float(fill.price)
        if quantity <= 0 or not math.isfinite(quantity) or price <= 0 or not math.isfinite(price):
            raise ValueError("fill quantity and price must be finite and positive")
        self.fills.append(fill)
        fee_rate = self.taker_fee if fill.is_aggressor else self.maker_fee
        fee = quantity * price * fee_rate
        signed_quantity = quantity if fill.side == Side.BUY else -quantity
        old_inventory = float(self.inventory)
        old_abs_inventory = abs(old_inventory)
        average_entry = self.cost_basis / old_abs_inventory if old_abs_inventory > 0 else 0.0
        new_inventory = old_inventory + signed_quantity

        self.cash -= signed_quantity * price + fee
        self.realized_pnl -= fee
        self.fees_paid += fee

        if old_inventory * signed_quantity < 0:
            close_quantity = min(old_abs_inventory, quantity)
            self.realized_pnl += close_quantity * (price - average_entry) * (1.0 if old_inventory > 0 else -1.0)
            remaining_old = old_abs_inventory - close_quantity
            remaining_new = quantity - close_quantity
            if remaining_old > 1e-12 and remaining_new <= 1e-12:
                self.cost_basis = average_entry * remaining_old
            elif remaining_new > 1e-12:
                self.cost_basis = remaining_new * price
            else:
                self.cost_basis = 0.0
        elif old_abs_inventory <= 1e-12 or old_inventory * signed_quantity > 0:
            self.cost_basis += quantity * price

        self.inventory = new_inventory


    def get_unrealized_pnl(self, current_price: float) -> float:
        """Return open-position mark-to-market P&L, excluding realized P&L."""
        if self.inventory > 0:
            return self.inventory * current_price - self.cost_basis
        if self.inventory < 0:
            return self.cost_basis - abs(self.inventory) * current_price
        return 0.0


    def get_total_pnl(self, current_price: float) -> float:
        """Return realized plus unrealized P&L net of all recorded fees."""
        return self.realized_pnl + self.get_unrealized_pnl(current_price)

    def get_state(self, current_price: float) -> StrategyState:
        """Get current strategy state."""
        return StrategyState(
            time=Timestamp(0.0),  # Set by caller
            inventory=self.inventory,
            cash=self.cash,
            unrealized_pnl=self.get_unrealized_pnl(current_price),
            realized_pnl=self.realized_pnl,
            total_fills=len(self.fills),
            active_orders=self.active_order_ids.copy(),
        )

    def reset(self) -> None:
        """Reset strategy state."""
        self.inventory = 0
        self.cash = 0.0
        self.realized_pnl = 0.0
        self.cost_basis = 0.0
        self.fees_paid = 0.0
        self.fills.clear()
        self.active_order_ids.clear()
