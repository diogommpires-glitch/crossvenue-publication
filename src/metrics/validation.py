"""Empirical validation metrics.

This module validates the simulator against known empirical laws:
1. Square-root law of market impact (critical for CrossVenue)
2. Adverse selection metrics

These validations demonstrate that the simulator produces realistic
market dynamics, not just plausible-looking order flow.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from scipy import stats

from ..orderbook.order import Fill
from ..utils.types import Side


@dataclass
class ImpactMeasurement:
    """Measurement of market impact for a meta-order."""

    quantity: int
    side: Side
    price_change: float  # Absolute price change
    fills: List[Fill]


class SquareRootLawValidator:
    """Validate simulator against square-root law of market impact.

    The square-root law states that market impact scales as:
        impact ~ sqrt(quantity)

    This is one of the most robust empirical relationships in market
    microstructure. Validating against it proves the simulator is realistic.

    Academic validation targets:
    - Log-log slope should be ~0.5 (theoretical)
    - Empirical slopes are typically 0.52-0.53
    - R² should be > 0.80 for high quality

    Reference: arXiv:2608.19227 (M3 model, 2026)
    """

    def __init__(self) -> None:
        self.measurements: List[ImpactMeasurement] = []

    def add_measurement(
        self,
        quantity: int,
        side: Side,
        price_before: float,
        price_after: float,
        fills: List[Fill],
    ) -> None:
        """Record a market impact measurement."""
        price_change = abs(price_after - price_before)

        measurement = ImpactMeasurement(
            quantity=quantity,
            side=side,
            price_change=price_change,
            fills=fills,
        )
        self.measurements.append(measurement)

    def validate(self) -> Dict:
        """Validate square-root law and return statistics.

        Returns:
        -------
        Dict with:
            - slope: Log-log regression slope (should be ~0.5)
            - intercept: Log-log regression intercept
            - r_squared: R² (should be > 0.8)
            - n_measurements: Number of data points
            - p_value: Statistical significance
        """
        if len(self.measurements) < 10:
            return {
                "slope": np.nan,
                "intercept": np.nan,
                "r_squared": np.nan,
                "n_measurements": len(self.measurements),
                "p_value": np.nan,
                "validation_status": "INSUFFICIENT_DATA",
            }

        # Extract quantities and impacts
        quantities = np.array([m.quantity for m in self.measurements])
        impacts = np.array([m.price_change for m in self.measurements])

        # Filter out zero impacts (no price movement)
        valid_mask = impacts > 0
        quantities = quantities[valid_mask]
        impacts = impacts[valid_mask]

        if len(quantities) < 10:
            return {
                "slope": np.nan,
                "intercept": np.nan,
                "r_squared": np.nan,
                "n_measurements": len(quantities),
                "p_value": np.nan,
                "validation_status": "INSUFFICIENT_NONZERO_DATA",
            }

        # Log-log regression
        log_qty = np.log(quantities)
        log_impact = np.log(impacts)

        # Linear regression on log-log data
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_qty, log_impact
        )

        r_squared = r_value ** 2

        # Determine validation status
        if r_squared > 0.80 and 0.4 < slope < 0.6:
            status = "PASS"
        elif r_squared > 0.70:
            status = "MARGINAL"
        else:
            status = "FAIL"

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "n_measurements": len(quantities),
            "p_value": p_value,
            "std_err": std_err,
            "validation_status": status,
        }

    def get_data_for_plotting(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get quantities and impacts for plotting.

        Returns:
        -------
        (quantities, impacts) arrays
        """
        if not self.measurements:
            return np.array([]), np.array([])

        quantities = np.array([m.quantity for m in self.measurements])
        impacts = np.array([m.price_change for m in self.measurements])

        return quantities, impacts


class AdverseSelectionMetrics:
    """Measure adverse selection in market-making fills.

    Adverse selection occurs when a market maker's fills are systematically
    worse than random - they buy before prices fall and sell before prices rise.

    Academic findings show 65-88% of real market-making fills are adverse.
    A simulator showing only profitable fills is unrealistic.
    """

    def __init__(self, lookforward_window: float = 10.0) -> None:
        """
        Parameters:
        ----------
        lookforward_window : float
            Time window (seconds) to measure price movement after fill
        """
        self.lookforward_window = lookforward_window
        self.fills_with_context: List[Tuple[Fill, float, float]] = []

    def record_fill(
        self,
        fill: Fill,
        price_at_fill: float,
        price_after: float,
    ) -> None:
        """Record a fill with before/after prices.

        Parameters:
        ----------
        fill : Fill
            The fill to record
        price_at_fill : float
            Mid-price at time of fill
        price_after : float
            Mid-price after lookforward_window
        """
        self.fills_with_context.append((fill, price_at_fill, price_after))

    def calculate_metrics(self) -> Dict:
        """Calculate adverse selection metrics.

        Returns:
        -------
        Dict with:
            - adverse_fill_ratio: % of fills that are adverse
            - avg_adverse_pnl: Average P&L on adverse fills
            - avg_favorable_pnl: Average P&L on favorable fills
            - total_fills: Number of fills analyzed
        """
        if not self.fills_with_context:
            return {
                "adverse_fill_ratio": np.nan,
                "avg_adverse_pnl": np.nan,
                "avg_favorable_pnl": np.nan,
                "total_fills": 0,
            }

        adverse_count = 0
        adverse_pnls = []
        favorable_pnls = []

        for fill, price_at_fill, price_after in self.fills_with_context:
            price_change = price_after - price_at_fill

            # For a BUY fill, adverse selection means price fell after
            # For a SELL fill, adverse selection means price rose after
            if fill.side == Side.BUY:
                # Bought, then price fell = adverse
                if price_change < 0:
                    adverse_count += 1
                    pnl = price_change * fill.quantity  # Negative
                    adverse_pnls.append(pnl)
                else:
                    pnl = price_change * fill.quantity  # Positive
                    favorable_pnls.append(pnl)

            else:  # SELL
                # Sold, then price rose = adverse
                if price_change > 0:
                    adverse_count += 1
                    pnl = -price_change * fill.quantity  # Negative
                    adverse_pnls.append(pnl)
                else:
                    pnl = -price_change * fill.quantity  # Positive
                    favorable_pnls.append(pnl)

        adverse_ratio = adverse_count / len(self.fills_with_context)

        return {
            "adverse_fill_ratio": adverse_ratio,
            "avg_adverse_pnl": np.mean(adverse_pnls) if adverse_pnls else 0.0,
            "avg_favorable_pnl": np.mean(favorable_pnls) if favorable_pnls else 0.0,
            "total_fills": len(self.fills_with_context),
        }
