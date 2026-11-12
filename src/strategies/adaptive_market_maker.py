"""Bounded fee-aware quoting candidate for offline research and explicit promotion."""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Dict

from .instrument_aware_market_maker import InstrumentAwareMarketMaker


class AdaptiveMarketMaker(InstrumentAwareMarketMaker):
    policy_version = "adaptive_v1"

    def __init__(self, *args, volatility_multiplier: float = 2.0,
                 adverse_selection_multiplier: float = 1.0,
                 max_spread_bps: float = 100.0, maker_per_unit: float = 0.0,
                 taker_per_unit: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        for name, value in (("volatility_multiplier", volatility_multiplier),
                            ("adverse_selection_multiplier", adverse_selection_multiplier),
                            ("max_spread_bps", max_spread_bps),
                            ("maker_per_unit", maker_per_unit), ("taker_per_unit", taker_per_unit)):
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be finite and non-negative".format(name))
            setattr(self, name, float(value))
        if self.max_spread_bps < self.spread_bps:
            raise ValueError("maximum spread cannot be below the reference spread")
        self.reference_spread_bps = self.spread_bps
        self._sample_time = None
        self._sample_mid = None
        self._variance = 0.0
        self._return_samples = 0
        self._pending_markouts = deque(maxlen=1000)
        self._adverse_markouts = deque(maxlen=100)
        self.last_policy_state = {}

    def observe_market_with_source(self, mid: float, receive_ts: float, event_ts: float) -> None:
        self.observe_market(mid, receive_ts, source_timestamp=event_ts)

    def observe_market(self, mid: float, timestamp: float, *, source_timestamp=None) -> None:
        mid, stamp = float(mid), float(timestamp)
        source_stamp = stamp if source_timestamp is None else float(source_timestamp)
        if not math.isfinite(mid) or mid <= 0 or not math.isfinite(stamp):
            return
        if not math.isfinite(source_stamp):
            return
        if self._sample_time is not None and stamp < self._sample_time:
            raise ValueError("adaptive observation time moved backwards")
        for entry in list(self._pending_markouts):
            age = stamp - entry["timestamp"]
            if age > 6:
                self._pending_markouts.remove(entry)
            elif age >= 5 and source_stamp >= entry["timestamp"] + 5 and entry["price"] != 0:
                self._pending_markouts.remove(entry)
                signed = entry["sign"] * (mid - entry["price"]) / abs(entry["price"]) * 10000
                self._adverse_markouts.append(max(0.0, -signed))
        if self._sample_time is None:
            self._sample_time, self._sample_mid = stamp, mid
        elif stamp - self._sample_time >= 1:
            elapsed = stamp - self._sample_time
            if elapsed <= 2:
                bps_return = math.log(mid / self._sample_mid) * 10000
                alpha = 1 - math.exp(-elapsed / 60.0)
                self._variance = (1 - alpha) * self._variance + alpha * bps_return ** 2
                self._return_samples += 1
            self._sample_time, self._sample_mid = stamp, mid

    def observe_fill(self, fill: Dict[str, Any]) -> None:
        self._pending_markouts.append({"timestamp": float(fill["timestamp"]),
            "price": float(fill["price"]), "sign": 1 if str(fill["side"]).upper() == "BUY" else -1})

    def on_market_event(self, market_state: Dict) -> list:
        mid = abs(float(market_state["mid_price"]))
        if mid == 0:
            return []
        # Full spread must cover a maker entry and conservative taker exit.
        unit_notional = mid * self.instrument.contract_multiplier
        fee_floor = ((self.maker_fee + self.taker_fee) * 10000
                     + (self.maker_per_unit + self.taker_per_unit) / unit_notional * 10000)
        tick_floor = self.instrument.tick_size / mid * 10000
        vol = math.sqrt(self._variance)
        adverse = sum(self._adverse_markouts) / len(self._adverse_markouts) if self._adverse_markouts else 0.0
        target = max(self.reference_spread_bps, tick_floor,
                     fee_floor + self.volatility_multiplier * vol + self.adverse_selection_multiplier * adverse)
        # If the cost floor is outside policy bounds, do not knowingly quote
        # below costs simply to satisfy an arbitrary maximum spread.
        if max(fee_floor, tick_floor) > self.max_spread_bps:
            self.last_policy_state = {"policy_version": self.policy_version, "ready": False,
                                      "reason": "cost_floor_exceeds_spread_bound"}
            return []
        self.spread_bps = min(self.max_spread_bps, target)
        self.last_policy_state = {"policy_version": self.policy_version, "ready": True,
            "spread_bps": self.spread_bps, "fee_floor_bps": fee_floor,
            "volatility_bps": vol, "adverse_selection_bps": adverse,
            "return_samples": self._return_samples,
            "calibrated": False, "execution_mode": "local_paper"}
        return super().on_market_event(market_state)


def adaptive_strategy_factory(parameters=None):
    values = dict(parameters or {})
    def factory(item):
        return AdaptiveMarketMaker(item.spec, order_size=item.order_size,
            max_inventory=item.max_position_quantity, spread_bps=item.spread_bps,
            inventory_skew_bps=item.inventory_skew_bps, maker_fee=item.fees.maker_rate,
            taker_fee=item.fees.taker_rate, maker_per_unit=item.fees.maker_per_unit,
            taker_per_unit=item.fees.taker_per_unit, strategy_id="mm:{}".format(item.spec.instrument_id), **values)
    return factory
