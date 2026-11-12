"""P&L tracking and performance metrics.

Tracks profit and loss, calculates risk-adjusted returns, and provides
diagnostic statistics for strategy evaluation.
"""

from dataclasses import dataclass
import math
from typing import List, Dict
import numpy as np

from ..orderbook.order import Fill
from ..utils.types import Side, Timestamp


@dataclass
class PnLSnapshot:
    """Point-in-time P&L snapshot."""

    timestamp: Timestamp
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    inventory: float
    cash: float


class PnLTracker:
    """Track P&L and calculate performance metrics.

    Maintains complete history of fills and P&L snapshots to compute:
    - Realized and unrealized P&L
    - Sharpe ratio (risk-adjusted returns)
    - Maximum drawdown
    - Win rate and profit factor
    """

    def __init__(
        self,
        strategy_id: str = "strategy",
        maker_fee: float = 0.001,
        taker_fee: float = 0.001,
        snapshot_interval: float = 60.0,  # seconds
        starting_capital: float = 10000.0,  # starting capital for return calculation
    ) -> None:
        if not math.isfinite(maker_fee) or not 0.0 <= maker_fee <= 0.1:
            raise ValueError("maker_fee must be finite and between 0 and 0.1")
        if not math.isfinite(taker_fee) or not 0.0 <= taker_fee <= 0.1:
            raise ValueError("taker_fee must be finite and between 0 and 0.1")
        if not math.isfinite(snapshot_interval) or snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be finite and positive")
        if not math.isfinite(starting_capital) or starting_capital <= 0:
            raise ValueError("starting_capital must be finite and positive")
        self.strategy_id = strategy_id
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.snapshot_interval = snapshot_interval
        self.starting_capital = starting_capital

        # State
        self.inventory: float = 0.0
        self.cash: float = 0.0
        self.realized_pnl: float = 0.0
        self.cost_basis: float = 0.0
        self.fees_paid: float = 0.0
        self.fills: List[Fill] = []
        self.snapshots: List[PnLSnapshot] = []
        self.fill_mid_prices: Dict[int, float] = {}  # fill index -> mid price at fill time

        # Running statistics
        self.max_pnl: float = 0.0
        self.max_drawdown: float = 0.0
        self.last_snapshot_time: float = 0.0

    def record_fill(
        self, fill: Fill, current_price: float, timestamp: Timestamp, mid_price: float
    ) -> None:
        """Apply a fill using average-cost accounting and explicit fees."""
        if fill.trader_id != self.strategy_id:
            return
        quantity = float(fill.quantity)
        price = float(fill.price)
        if quantity <= 0 or not math.isfinite(quantity) or price <= 0 or not math.isfinite(price):
            raise ValueError("fill quantity and price must be finite and positive")

        fill_idx = len(self.fills)
        self.fills.append(fill)
        self.fill_mid_prices[fill_idx] = float(mid_price)
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
        self._maybe_take_snapshot(current_price, timestamp)


    def _maybe_take_snapshot(self, current_price: float, timestamp: Timestamp) -> None:
        """Take snapshot on regular time grid."""
        timestamp_val = float(timestamp) if not isinstance(timestamp, float) else timestamp
        if timestamp_val - self.last_snapshot_time >= self.snapshot_interval:
            self._take_snapshot(current_price, timestamp)
            self.last_snapshot_time = timestamp_val

    def _take_snapshot(self, current_price: float, timestamp: Timestamp) -> None:
        """Take a mark-to-market snapshot without double-counting realized P&L."""
        current = self.get_current_pnl(current_price)
        total = current["total_pnl"]
        snapshot = PnLSnapshot(
            timestamp=timestamp,
            realized_pnl=current["realized_pnl"],
            unrealized_pnl=current["unrealized_pnl"],
            total_pnl=total,
            inventory=self.inventory,
            cash=self.cash,
        )
        self.snapshots.append(snapshot)
        self.max_pnl = max(self.max_pnl, total)
        self.max_drawdown = max(self.max_drawdown, self.max_pnl - total)


    def get_current_pnl(self, current_price: float) -> Dict[str, float]:
        """Return realized, unrealized, and total net P&L in quote currency."""
        price = float(current_price)
        if not math.isfinite(price) or price < 0:
            raise ValueError("current_price must be finite and non-negative")
        if self.inventory > 0:
            unrealized = self.inventory * price - self.cost_basis
        elif self.inventory < 0:
            unrealized = self.cost_basis - abs(self.inventory) * price
        else:
            unrealized = 0.0
        total = self.cash + self.inventory * price
        return {
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized,
            "total_pnl": total,
            "inventory": self.inventory,
            "cash": self.cash,
            "fees_paid": self.fees_paid,
        }


    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from regular time-grid snapshots.

        Sharpe = (mean_return - risk_free_rate) / std_return

        Uses snapshots taken at regular intervals (not fill times) to compute
        proper time-series returns. Returns are calculated as equity changes
        divided by starting capital, not previous P&L.

        Returns NaN if insufficient data, inf/-inf for zero variance with
        positive/negative mean returns.
        """
        if len(self.snapshots) < 2:
            return np.nan

        # Calculate returns from time-grid snapshots
        # Returns = (equity_change) / starting_capital
        pnls = np.array([s.total_pnl for s in self.snapshots])
        equity_changes = np.diff(pnls)
        returns = equity_changes / self.starting_capital

        if len(returns) == 0:
            return np.nan

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)  # Sample std with ddof=1

        # Handle zero variance case
        if std_return == 0 or np.isclose(std_return, 0):
            if mean_return > risk_free_rate:
                return np.inf
            elif mean_return < risk_free_rate:
                return -np.inf
            else:
                return np.nan

        sharpe = (mean_return - risk_free_rate) / std_return

        # Annualize: snapshots are at snapshot_interval seconds apart
        # Convert to annual periods
        periods_per_year = (365.25 * 24 * 3600) / self.snapshot_interval
        sharpe_annualized = sharpe * np.sqrt(periods_per_year)

        return sharpe_annualized

    def calculate_win_rate(self) -> float:
        """Calculate adverse selection metric.

        Measures percentage of fills that were profitable relative to mid-price
        at fill time. A buy is "good" if fill price < mid, a sell is "good" if
        fill price > mid. This captures whether the strategy is getting adverse
        selection from informed traders.
        """
        if not self.fills:
            return 0.0

        wins = 0
        for idx, fill in enumerate(self.fills):
            mid = self.fill_mid_prices.get(idx, fill.price)

            if fill.side == Side.BUY and fill.price < mid:
                wins += 1
            elif fill.side == Side.SELL and fill.price > mid:
                wins += 1

        return wins / len(self.fills) if self.fills else 0.0

    def get_summary(self, current_price: float) -> Dict:
        """Get comprehensive performance summary."""
        current_pnl = self.get_current_pnl(current_price)
        sharpe = self.calculate_sharpe_ratio()
        win_rate = self.calculate_win_rate()

        return {
            "total_pnl": current_pnl["total_pnl"],
            "realized_pnl": current_pnl["realized_pnl"],
            "unrealized_pnl": current_pnl["unrealized_pnl"],
            "sharpe_ratio": sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": win_rate,
            "total_fills": len(self.fills),
            "final_inventory": self.inventory,
        }

    def reset(self) -> None:
        """Reset balances, fills, fee totals, and performance history."""
        self.inventory = 0.0
        self.cash = 0.0
        self.realized_pnl = 0.0
        self.cost_basis = 0.0
        self.fills.clear()
        self.snapshots.clear()
        self.fill_mid_prices.clear()
        self.fees_paid = 0.0
        self.max_pnl = 0.0
        self.max_drawdown = 0.0
        self.last_snapshot_time = 0.0

