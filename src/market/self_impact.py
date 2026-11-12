"""Signed transient price impact from aggressive order flow.

The model uses a linear response in trade size relative to pre-trade visible
depth, followed by exponential resilience. It tracks strategy flow separately
from other aggressive flow while composing both into the simulated market price.
"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Dict, Optional

import numpy as np

from ..orderbook.order import Fill
from ..utils.types import Price, Side, Timestamp


@dataclass
class ImpactMeasurement:
    """Record a measured trade and its observed price response."""

    timestamp: Timestamp
    side: Side
    quantity: float
    price_before: Price
    price_after: Price
    impact: float

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")


@dataclass(frozen=True)
class _ImpactImpulse:
    timestamp: float
    log_impact: float
    source: str


@dataclass
class _OrderProgress:
    side: Side
    source: str
    quantity: float
    reference_quantity: float
    last_fill_time: float


class SelfImpactModel:
    """Track aggressor impact and decay it against elapsed simulation time.

    impact_coefficient_bps is the log-price response in basis points for an
    aggressive order equal to reference_quantity. The simulator normally
    supplies pre-trade displayed depth for the incoming order side. The
    configured reference quantity is used only when that depth is empty.

    Partial fills sharing an order ID are accumulated before calculating the
    linear order-size response. Passive fills are recorded for diagnostics but
    do not create a price impulse.
    """

    MAX_RELATIVE_IMPACT = 0.10

    def __init__(
        self,
        impact_coefficient_bps: float = 5.0,
        reference_quantity: float = 100.0,
        decay_halflife: float = 30.0,
        strategy_id: str = "mm_strategy",
        lookback_window: float = 300.0,
    ) -> None:
        if not math.isfinite(impact_coefficient_bps) or impact_coefficient_bps < 0:
            raise ValueError("impact_coefficient_bps must be finite and non-negative")
        if not math.isfinite(reference_quantity) or reference_quantity <= 0:
            raise ValueError("reference_quantity must be finite and positive")
        if not math.isfinite(decay_halflife) or decay_halflife <= 0:
            raise ValueError("decay_halflife must be finite and positive")
        if not math.isfinite(lookback_window) or lookback_window <= 0:
            raise ValueError("lookback_window must be finite and positive")

        self.impact_coefficient_bps = float(impact_coefficient_bps)
        self.reference_quantity = float(reference_quantity)
        self.decay_halflife = float(decay_halflife)
        self.strategy_id = strategy_id
        self.lookback_window = float(lookback_window)

        self._strategy_fills: Deque[Fill] = deque()
        self._all_fills: Deque[Fill] = deque()
        self._impulses: Deque[_ImpactImpulse] = deque()
        self._order_progress: Dict[str, _OrderProgress] = {}
        self._last_update: Optional[Timestamp] = None
        self._state_time: Optional[float] = None
        self._last_strategy_impulse: Optional[float] = None
        self._last_market_impulse: Optional[float] = None

        # These are relative log-price displacements, not dollar changes.
        self._cumulative_impact = 0.0
        self._strategy_impact = 0.0
        self._market_impact = 0.0

    def advance_time(self, current_time: Timestamp) -> None:
        """Advance impact decay without requiring a new fill."""
        now = float(current_time)
        if not math.isfinite(now):
            raise ValueError("current_time must be finite")
        if self._last_update is not None and now < float(self._last_update):
            raise ValueError(
                f"Impact time moved backwards: {now} < {float(self._last_update)}"
            )

        self._last_update = Timestamp(now)
        self._cleanup_old_fills(Timestamp(now))
        self._refresh_impact_state(now)

    def record_fill(
        self,
        fill: Fill,
        current_time: Timestamp,
        reference_quantity: Optional[float] = None,
    ) -> None:
        """Record a fill and add one impulse for its aggressive order.

        reference_quantity should be the visible depth consumed, measured
        immediately before matching.
        """
        self.advance_time(current_time)
        self._all_fills.append(fill)

        is_strategy_fill = fill.trader_id == self.strategy_id
        if is_strategy_fill:
            self._strategy_fills.append(fill)

        # Passive fills are caused by the counterparty aggressor. Its matching
        # Fill record carries the market-pressure direction.
        if not fill.is_aggressor:
            self._cleanup_old_fills(current_time)
            self._refresh_impact_state(float(current_time))
            return

        source = "strategy" if is_strategy_fill else "market"
        now = float(current_time)
        previous = self._order_progress.get(fill.order_id)
        if (
            previous is None
            or previous.side != fill.side
            or previous.source != source
            or now - previous.last_fill_time > self.lookback_window
        ):
            previous_quantity = 0.0
            order_reference = self._valid_reference(reference_quantity)
        else:
            previous_quantity = previous.quantity
            order_reference = previous.reference_quantity

        new_quantity = previous_quantity + float(fill.quantity)
        old_response = self._response(previous_quantity, order_reference)
        new_response = self._response(new_quantity, order_reference)
        signed_impulse = new_response - old_response
        if fill.side == Side.SELL:
            signed_impulse = -signed_impulse

        if signed_impulse != 0.0:
            self._impulses.append(
                _ImpactImpulse(timestamp=now, log_impact=signed_impulse, source=source)
            )
            lower_cap, upper_cap = self._log_bounds
            if source == "strategy":
                self._strategy_impact = float(np.clip(
                    self._strategy_impact + signed_impulse, lower_cap, upper_cap
                ))
                self._last_strategy_impulse = now
            else:
                self._market_impact = float(np.clip(
                    self._market_impact + signed_impulse, lower_cap, upper_cap
                ))
                self._last_market_impulse = now
            self._cumulative_impact = float(np.clip(
                self._strategy_impact + self._market_impact, lower_cap, upper_cap
            ))

        self._order_progress[fill.order_id] = _OrderProgress(
            side=fill.side,
            source=source,
            quantity=new_quantity,
            reference_quantity=order_reference,
            last_fill_time=now,
        )
        self._cleanup_old_fills(current_time)
        self._refresh_impact_state(now)

    def get_impact_adjusted_price(self, base_mid_price: Price) -> Price:
        """Compose one market price from the unaffected price and impact."""
        if base_mid_price <= 0:
            return base_mid_price
        relative_impact = self.get_relative_impact(base_mid_price)
        return Price(float(base_mid_price) * (1.0 + relative_impact))

    def get_cumulative_impact(self, current_time: Optional[Timestamp] = None) -> float:
        """Return total relative log-price impact at the current model time."""
        if current_time is not None:
            self.advance_time(current_time)
        return self._cumulative_impact

    def get_strategy_impact(self, current_time: Optional[Timestamp] = None) -> float:
        """Return the decayed log-price contribution from strategy aggression."""
        if current_time is not None:
            self.advance_time(current_time)
        return self._strategy_impact

    def get_market_impact(self, current_time: Optional[Timestamp] = None) -> float:
        """Return the decayed contribution from other aggressive orders."""
        if current_time is not None:
            self.advance_time(current_time)
        return self._market_impact

    def get_relative_impact(self, current_price: Price) -> float:
        """Return the bounded relative price change implied by current impact."""
        if current_price <= 0:
            return 0.0

        min_log, max_log = self._log_bounds
        bounded_log_impact = float(np.clip(self._cumulative_impact, min_log, max_log))
        return math.expm1(bounded_log_impact)

    def get_strategy_fill_count(self) -> int:
        """Return strategy fills in the retained lookback window."""
        return len(self._strategy_fills)

    def get_total_fill_count(self) -> int:
        """Return all fill records in the retained lookback window."""
        return len(self._all_fills)

    def estimate_additional_impact(
        self,
        quantity: float,
        side: Side,
        reference_quantity: Optional[float] = None,
    ) -> float:
        """Estimate the signed log-price impulse for one aggressive order."""
        if not math.isfinite(quantity) or quantity <= 0:
            raise ValueError("quantity must be finite and positive")
        reference = self._valid_reference(reference_quantity)
        impact = self._response(float(quantity), reference)
        return impact if side == Side.BUY else -impact

    def get_statistics(self) -> Dict[str, float]:
        """Return impact and calibration settings for reports."""
        raw_strategy = sum(
            impulse.log_impact * self._decay_factor(float(self._last_update or 0.0) - impulse.timestamp)
            for impulse in self._impulses if impulse.source == "strategy"
        )
        raw_market = sum(
            impulse.log_impact * self._decay_factor(float(self._last_update or 0.0) - impulse.timestamp)
            for impulse in self._impulses if impulse.source == "market"
        )
        raw_total = raw_strategy + raw_market
        return {
            "cumulative_impact_log_return": self._cumulative_impact,
            "strategy_impact_log_return": self._strategy_impact,
            "market_impact_log_return": self._market_impact,
            "cumulative_impact_bps": math.expm1(self._cumulative_impact) * 10000.0,
            "raw_cumulative_impact_log_return": raw_total,
            "raw_cumulative_impact_bps": math.expm1(float(np.clip(raw_total, -700.0, 700.0))) * 10000.0,
            "max_applied_relative_impact": self.MAX_RELATIVE_IMPACT,
            "strategy_fills": float(len(self._strategy_fills)),
            "total_fills": float(len(self._all_fills)),
            "impact_coefficient_bps": self.impact_coefficient_bps,
            "reference_quantity": self.reference_quantity,
            "decay_halflife": self.decay_halflife,
            "lookback_window": self.lookback_window,
        }

    def reset(self) -> None:
        """Clear fills, impulses, and state for a fresh simulation."""
        self._strategy_fills.clear()
        self._all_fills.clear()
        self._impulses.clear()
        self._order_progress.clear()
        self._last_update = None
        self._state_time = None
        self._last_strategy_impulse = None
        self._last_market_impulse = None
        self._cumulative_impact = 0.0
        self._strategy_impact = 0.0
        self._market_impact = 0.0

    def _valid_reference(self, reference_quantity: Optional[float]) -> float:
        if reference_quantity is None:
            return self.reference_quantity
        if not math.isfinite(reference_quantity):
            raise ValueError("reference_quantity must be finite")
        if reference_quantity <= 0:
            return self.reference_quantity
        return float(reference_quantity)

    def _response(self, quantity: float, reference_quantity: float) -> float:
        # Linear response is the baseline for exponential decay.
        return (self.impact_coefficient_bps / 10000.0) * (
            quantity / reference_quantity
        )

    def _decay_factor(self, age: float) -> float:
        return math.exp(-math.log(2.0) * max(0.0, age) / self.decay_halflife)

    @property
    def _log_bounds(self) -> tuple[float, float]:
        return math.log1p(-self.MAX_RELATIVE_IMPACT), math.log1p(self.MAX_RELATIVE_IMPACT)

    def _refresh_impact_state(self, current_time: float) -> None:
        if self._state_time is None:
            self._state_time = current_time
        else:
            factor = self._decay_factor(current_time - self._state_time)
            self._strategy_impact *= factor
            self._market_impact *= factor
            self._state_time = current_time
        if (self._last_strategy_impulse is not None
                and current_time - self._last_strategy_impulse > self.lookback_window):
            self._strategy_impact = 0.0
            self._last_strategy_impulse = None
        if (self._last_market_impulse is not None
                and current_time - self._last_market_impulse > self.lookback_window):
            self._market_impact = 0.0
            self._last_market_impulse = None
        lower_cap, upper_cap = self._log_bounds
        self._cumulative_impact = float(np.clip(
            self._strategy_impact + self._market_impact, lower_cap, upper_cap
        ))

    def _cleanup_old_fills(self, current_time: Timestamp) -> None:
        now = float(current_time)
        cutoff = now - self.lookback_window
        while self._strategy_fills and float(self._strategy_fills[0].timestamp) < cutoff:
            self._strategy_fills.popleft()
        while self._all_fills and float(self._all_fills[0].timestamp) < cutoff:
            self._all_fills.popleft()
        while self._impulses and self._impulses[0].timestamp < cutoff:
            self._impulses.popleft()

        expired = [
            order_id
            for order_id, progress in self._order_progress.items()
            if progress.last_fill_time < cutoff
        ]
        for order_id in expired:
            del self._order_progress[order_id]

    def __str__(self) -> str:
        bounded_log = float(np.clip(self._cumulative_impact, math.log1p(-self.MAX_RELATIVE_IMPACT), math.log1p(self.MAX_RELATIVE_IMPACT)))
        impact_bps = math.expm1(bounded_log) * 10000.0
        return (
            f"SelfImpactModel(impact={impact_bps:.2f} bps, "
            f"strategy_fills={len(self._strategy_fills)}, "
            f"all_fills={len(self._all_fills)})"
        )
