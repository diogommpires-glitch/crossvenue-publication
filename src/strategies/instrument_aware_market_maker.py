"""Instrument-aware two-sided quoting in explicit tick, lot, and exposure units.

This deliberately small strategy is the cross-asset companion to the legacy
Avellaneda--Stoikov implementation. It quotes from normalized market state and
keeps account accounting in the portfolio ledger, so futures contract notionals
and FX cash balances are not approximated as spot inventory cash flows.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math
from typing import Dict, List, Optional

from ..market.instruments import InstrumentSpec
from ..orderbook.order import Fill, Order
from ..utils.types import OrderType, Price, Quantity, Side, Timestamp


def _round_step(value: float, step: float, rounding: str) -> float:
    """Round a finite value to a positive step without binary-float tick drift."""
    if not math.isfinite(value) or not math.isfinite(step) or step <= 0:
        raise ValueError("value must be finite and step must be finite and positive")
    mode = ROUND_FLOOR if rounding == "down" else ROUND_CEILING
    units = (Decimal(str(value)) / Decimal(str(step))).to_integral_value(rounding=mode)
    return float(units * Decimal(str(step)))


class InstrumentAwareMarketMaker:
    """Two-sided quote generator whose parameters are expressed per instrument.

    ``spread_bps`` is the full target quote spread before tick rounding.
    ``inventory_skew_bps`` is the reservation-price shift at the configured
    maximum inventory. Quantity fields use the instrument's ``quantity_unit``.
    """

    def __init__(
        self,
        instrument: InstrumentSpec,
        *,
        order_size: float,
        max_inventory: float,
        spread_bps: float = 8.0,
        inventory_skew_bps: float = 8.0,
        strategy_id: Optional[str] = None,
        maker_fee: float = 0.0,
        taker_fee: float = 0.0,
        allow_short: bool = True,
    ) -> None:
        self.instrument = instrument
        self.strategy_id = strategy_id or f"mm:{instrument.instrument_id}"
        self.order_size = self._positive("order_size", order_size)
        self.max_inventory = self._positive("max_inventory", max_inventory)
        self.spread_bps = self._nonnegative("spread_bps", spread_bps)
        self.inventory_skew_bps = self._nonnegative("inventory_skew_bps", inventory_skew_bps)
        self.maker_fee = self._nonnegative("maker_fee", maker_fee)
        self.taker_fee = self._nonnegative("taker_fee", taker_fee)
        self.allow_short = allow_short
        self.inventory = 0.0
        self.fills: List[Fill] = []
        self.active_order_ids: List[str] = []
        if self.order_size + 1e-12 < float(instrument.min_quantity):
            raise ValueError("order_size is below the instrument minimum quantity")

    @staticmethod
    def _positive(name: str, value: float) -> float:
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    @staticmethod
    def _nonnegative(name: str, value: float) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    def on_market_event(self, market_state: Dict) -> List[Order]:
        """Generate resting limit quotes from one instrument's current market."""
        if market_state.get("instrument_id", self.instrument.instrument_id) != self.instrument.instrument_id:
            raise ValueError("market event instrument does not match strategy instrument")
        if not market_state.get("market_open", True) or not market_state.get("market_data_ready", True):
            return []

        mid = float(market_state["mid_price"])
        timestamp = float(market_state["time"])
        if not math.isfinite(mid) or (mid <= 0 and self.instrument.asset_class.value != "future") or not math.isfinite(timestamp):
            return []
        tick = float(self.instrument.tick_size)
        lot = float(self.instrument.quantity_step)
        size = _round_step(self.order_size, lot, "down")
        if size + 1e-12 < float(self.instrument.min_quantity):
            return []

        inventory_ratio = max(-1.0, min(1.0, self.inventory / self.max_inventory))
        reservation = mid - abs(mid) * inventory_ratio * self.inventory_skew_bps / 10_000.0
        half_spread = max(abs(mid) * self.spread_bps / 20_000.0, tick / 2.0)
        bid_price = self.instrument.round_price(reservation - half_spread, Side.BUY)
        ask_price = self.instrument.round_price(reservation + half_spread, Side.SELL)
        if (self.instrument.asset_class.value != "future" and bid_price <= 0) or ask_price <= bid_price:
            return []

        can_buy = self.inventory + size <= self.max_inventory + 1e-12
        sell_size = size
        if not self.allow_short:
            owned = max(0.0, self.inventory)
            next_lot = _round_step(owned, lot, "up")
            # Filled lots can leave subtraction roundoff below the next lot boundary.
            roundoff = min(4 * math.ulp(self.max_inventory), lot * 1e-12)
            if next_lot - owned <= roundoff:
                owned = next_lot
            sell_size = _round_step(min(size, owned), lot, "down")
        min_inventory = -self.max_inventory if self.allow_short else 0.0
        can_sell = (sell_size > 0 and sell_size >= float(self.instrument.min_quantity)
                    and self.inventory - sell_size >= min_inventory - 1e-12)
        orders = []
        if can_buy:
            orders.append(Order(
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                quantity=Quantity(size),
                price=Price(bid_price),
                timestamp=Timestamp(timestamp),
                trader_id=self.strategy_id,
                instrument_id=self.instrument.instrument_id,
            ))
        if can_sell:
            orders.append(Order(
                side=Side.SELL,
                order_type=OrderType.LIMIT,
                quantity=Quantity(sell_size),
                price=Price(ask_price),
                timestamp=Timestamp(timestamp),
                trader_id=self.strategy_id,
                instrument_id=self.instrument.instrument_id,
            ))
        return orders

    def on_fill(self, fill: Fill, current_price: Optional[float] = None) -> None:
        """Update only strategy inventory; the portfolio ledger owns all money."""
        if fill.instrument_id not in (None, self.instrument.instrument_id):
            raise ValueError("fill instrument does not match strategy instrument")
        signed = float(fill.quantity) if fill.side == Side.BUY else -float(fill.quantity)
        self.inventory += signed
        self.fills.append(fill)

    def sync_inventory(self, position: float) -> None:
        value = float(position)
        if not math.isfinite(value):
            raise ValueError("position must be finite")
        self.inventory = value

    def reset(self) -> None:
        self.inventory = 0.0
        self.fills.clear()
        self.active_order_ids.clear()
