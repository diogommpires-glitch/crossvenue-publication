"""Durable portfolio observations and explicitly defined paper research metrics."""

from __future__ import annotations

from collections import deque
import math
import statistics
from typing import Any, Dict, Optional

from ..market.events import MarketDataEvent, MarketEventKind


class PortfolioAnalytics:
    """One-second history, complete-minute returns, and observed fill markouts.

    Return intervals are time-weighted and adjusted for external funding. A
    missing/invalid observation breaks the interval instead of filling gaps.
    Annualization is disabled unless the caller supplies an explicit factor.
    """

    def __init__(self, state_manager: Any = None, account_id: str = "paper", *,
                 sample_interval: float = 1.0, periods_per_year: Optional[float] = None,
                 max_mark_age: float = 10.0):
        if sample_interval <= 0 or max_mark_age <= 0:
            raise ValueError("sample interval and mark age must be positive")
        if periods_per_year is not None and (not math.isfinite(periods_per_year) or periods_per_year <= 0):
            raise ValueError("periods_per_year must be positive")
        self.state_manager = state_manager
        self.account_id = account_id
        self.sample_interval = float(sample_interval)
        self.periods_per_year = periods_per_year
        self.max_mark_age = max_mark_age
        self._history = deque(maxlen=3600)
        self._returns = deque(maxlen=10080)
        self._previous = None
        self._minute_start = None
        self._minute_factor = 1.0
        self._minute_valid = True
        self._minute_count = 0
        self._wealth_index = self._wealth_peak = 1.0
        self._max_drawdown = 0.0
        self._pending = {}
        self._markouts = deque(maxlen=10000)
        self._unpersisted_markouts = []
        self._latest_summary = {}
        self._pending_flow = 0.0
        self._funding_ids = set()
        self._restore_history()

    def _restore_history(self) -> None:
        if self.state_manager is None:
            return
        pages = []
        batch = self.state_manager.get_account_history(self.account_id, limit=100000)
        if not batch:
            return
        end = float(batch[-1]["timestamp"])
        minimum = end - 7 * 86400
        pages.append(batch)
        while len(batch) == 100000 and float(batch[0]["timestamp"]) > minimum:
            earlier = math.nextafter(float(batch[0]["timestamp"]), -math.inf)
            batch = self.state_manager.get_account_history(self.account_id, start=minimum, end=earlier, limit=100000)
            if not batch:
                break
            pages.append(batch)
        rows = [row for page in reversed(pages) for row in page if row["timestamp"] >= minimum]
        for index, raw in enumerate(rows):
            row = dict(raw.get("payload", raw))
            row["timestamp"] = float(raw.get("timestamp", row["timestamp"]))
            row.setdefault("external_flow", 0.0)
            row.setdefault("valuation_valid", row.get("equity") is not None)
            state = row.get("analytics_state")
            if index == 0 and state:
                self._wealth_index = float(state["wealth_index"])
                self._wealth_peak = float(state["wealth_peak"])
                self._max_drawdown = float(state["max_drawdown"])
                self._minute_start = state["minute_start"]
                self._minute_factor = float(state["minute_factor"])
                self._minute_valid = bool(state["minute_valid"])
                self._minute_count = int(state["minute_count"])
                if row.get("minute_return") is not None:
                    self._returns.append({"timestamp": row["timestamp"], "return": row["minute_return"]})
            else:
                self._update_returns(row)
            for markout in row.get("markout_observations", ()):
                self._markouts.append(markout)
            self._history.append(row)
            self._previous = row
            self._latest_summary = row
        getter = getattr(self.state_manager, "get_cross_asset_events", None)
        after = getattr(self.state_manager, "get_account_events_after", None)
        revision = self._previous.get("account_revision") if self._previous is not None else None
        if callable(after) and isinstance(revision, int):
            # A deposit can commit after the last sample and before a crash.
            # Its economic date may be backdated, so only journal order can
            # identify the flow that has not yet entered return calculation.
            for event in after(self.account_id, revision, event_type="funding"):
                self.on_funding(event["payload"])
        if callable(getter) and self._previous is not None:
            cutoff = self._previous["timestamp"] - 31
            observed = {(item["fill_id"], item["horizon_seconds"]) for item in self._markouts}
            for journal_event in getter(self.account_id, limit=2000):
                payload = journal_event.get("payload", {})
                fills = [payload] if journal_event.get("event_type") == "fill" else payload.get("legs", [])
                for fill in fills:
                    if float(fill.get("timestamp", -math.inf)) >= cutoff and "instrument_id" in fill:
                        self.on_fill(fill)
            for key, fill in self._pending.items():
                fill["observed"] = {horizon for horizon in (1, 5, 30) if (key, horizon) in observed}

    def on_funding(self, event: Dict[str, Any]) -> None:
        identity = event.get("funding_id", event.get("event_id"))
        if identity is not None and identity in self._funding_ids:
            return
        amount = event.get("external_flow_reporting")
        if amount is None or not math.isfinite(float(amount)):
            raise ValueError("funding analytics requires explicit reporting-currency flow")
        self._pending_flow += float(amount)
        if identity is not None:
            self._funding_ids.add(identity)
            if len(self._funding_ids) > 4096:
                self._funding_ids = {identity}

    def on_fill(self, fill: Dict[str, Any]) -> None:
        identity = fill.get("fill_id") or "{}:{}:{}".format(fill.get("order_id"), fill.get("timestamp"), fill.get("quantity"))
        if identity in self._pending:
            return
        self._pending[identity] = {"fill_id": identity, "instrument_id": fill["instrument_id"],
            "timestamp": float(fill["timestamp"]), "price": float(fill["price"]),
            "side": str(fill["side"]).upper(), "observed": set()}

    def on_market(self, event: MarketDataEvent) -> None:
        if event.kind != MarketEventKind.BOOK_SNAPSHOT or not event.bids or not event.asks:
            return
        if event.bids[0][0] >= event.asks[0][0]:
            return
        if not 0 <= event.receive_ts - event.event_ts <= self.max_mark_age:
            return
        mid = (event.bids[0][0] + event.asks[0][0]) / 2
        for key, fill in list(self._pending.items()):
            if fill["instrument_id"] != event.instrument_id:
                continue
            age = event.receive_ts - fill["timestamp"]
            for horizon in (1, 5, 30):
                if horizon in fill["observed"]:
                    continue
                # Accept only the first quote within one second after due time;
                # a quote arriving much later does not establish that markout.
                if age > horizon + 1:
                    fill["observed"].add(horizon)
                elif age >= horizon and event.event_ts >= fill["timestamp"] + horizon:
                    fill["observed"].add(horizon)
                    sign = 1 if fill["side"] == "BUY" else -1
                    bps = None if fill["price"] == 0 else sign * (mid - fill["price"]) / abs(fill["price"]) * 10000
                    observation = {"fill_id": key, "instrument_id": event.instrument_id,
                            "horizon_seconds": horizon, "observed_timestamp": event.receive_ts,
                            "source_timestamp": event.event_ts, "mid_price": mid,
                            "signed_markout_bps": bps, "execution_mode": "local_paper"}
                    self._markouts.append(observation)
                    self._unpersisted_markouts.append(observation)
            if age > 31 or len(fill["observed"]) == 3:
                self._pending.pop(key, None)
        # Bound pending diagnostics independently of permanent fill history.
        while len(self._pending) > 10000:
            self._pending.pop(next(iter(self._pending)))

    def observe(self, timestamp: float, portfolio: Optional[Dict[str, Any]], instruments=None,
                external_flow: float = 0.0) -> Dict[str, Any]:
        stamp = float(timestamp)
        if not math.isfinite(stamp) or not math.isfinite(float(external_flow)):
            raise ValueError("observation time and external flow must be finite")
        self._pending_flow += float(external_flow)
        if self._previous and stamp < self._previous["timestamp"]:
            raise ValueError("analytics time cannot move backwards")
        if self._previous and stamp - self._previous["timestamp"] < self.sample_interval:
            return self.summary()
        data = dict(portfolio or {})
        instruments = list(instruments or ())
        equity = data.get("equity")
        valid = (equity is not None and math.isfinite(float(equity)) and float(equity) > 0
                 and data.get("valuation_valid", True) is not False)
        row = {"timestamp": stamp, "valuation_valid": valid,
               "account_revision": data.get("account_revision"),
               "reporting_currency": data.get("reporting_currency", "USD"),
               "equity": float(equity) if valid else None,
               "total_pnl": data.get("total_pnl"), "external_flow": self._pending_flow,
               "positions": {item["instrument_id"]: item.get("position", 0) for item in instruments},
               "fees_paid": data.get("fees_paid"), "costs_paid": data.get("costs_paid"),
               "income": data.get("income"), "translation_pnl": data.get("translation_pnl"),
               "initial_margin_required": data.get("initial_margin_required"),
               "maintenance_margin_required": data.get("maintenance_margin_required"),
               "valuation_error": data.get("valuation_error")}
        gross = {}
        net = {}
        for item in instruments:
            mid = item.get("mid_price")
            if mid is None:
                continue
            currency = item.get("settlement_currency") if item.get("asset_class") == "future" else item.get("quote_currency")
            if not currency:
                continue
            value = float(item.get("position", 0)) * float(mid) * float(item.get("contract_multiplier", 1))
            gross[currency] = gross.get(currency, 0) + abs(value)
            net[currency] = net.get(currency, 0) + value
        rates = dict(data.get("fx_rates", {}))
        rates[row["reporting_currency"]] = 1.0
        rates_valid = all(currency in rates and rates[currency] is not None and rates[currency] > 0 for currency in gross)
        row.update({"gross_exposure_by_currency": gross, "net_exposure_by_currency": net,
                    "gross_exposure": sum(value * rates[currency] for currency, value in gross.items()) if rates_valid else None,
                    "net_exposure": sum(value * rates[currency] for currency, value in net.items()) if rates_valid else None,
                    "currency_balances": data.get("cash_by_currency", {}),
                    "margin_headroom": (float(equity) - float(data.get("initial_margin_required", 0))) if valid else None})
        self._update_returns(row)
        row["analytics_state"] = {"wealth_index": self._wealth_index, "wealth_peak": self._wealth_peak,
            "max_drawdown": self._max_drawdown, "minute_start": self._minute_start,
            "minute_factor": self._minute_factor, "minute_valid": self._minute_valid,
            "minute_count": self._minute_count}
        row["minute_return"] = self._returns[-1]["return"] if self._returns and self._returns[-1]["timestamp"] == stamp else None
        row["markout_observations"] = list(self._unpersisted_markouts)
        if self.state_manager is not None:
            self.state_manager.record_account_observation(self.account_id, stamp, row)
        self._unpersisted_markouts.clear()
        self._history.append(row)
        self._previous = row
        self._pending_flow = 0.0
        self._latest_summary = row
        return self.summary()

    def _update_returns(self, row):
        previous = self._previous
        if previous is None:
            self._minute_start = row["timestamp"]
            self._minute_valid = row["valuation_valid"]
            return
        gap = row["timestamp"] - previous["timestamp"]
        good = (row["valuation_valid"] and previous["valuation_valid"]
                and 0 < gap <= self.sample_interval * 1.5)
        if good:
            factor = (row["equity"] - row["external_flow"]) / previous["equity"]
            if factor <= 0:
                good = False
            else:
                self._minute_factor *= factor
                self._wealth_index *= factor
                self._wealth_peak = max(self._wealth_peak, self._wealth_index)
                self._max_drawdown = max(self._max_drawdown, 1 - self._wealth_index / self._wealth_peak)
                self._minute_count += 1
        self._minute_valid = self._minute_valid and good
        if row["timestamp"] - self._minute_start >= 60:
            duration = row["timestamp"] - self._minute_start
            if self._minute_valid and duration <= 60 + self.sample_interval * 0.5 and self._minute_count >= int(60 / self.sample_interval):
                self._returns.append({"timestamp": row["timestamp"], "return": self._minute_factor - 1})
            self._minute_start = row["timestamp"]
            self._minute_factor = 1.0
            self._minute_valid = row["valuation_valid"]
            self._minute_count = 0

    def history(self, start=None, end=None, limit: int = 1000):
        if self.state_manager is not None:
            return self.state_manager.get_account_history(self.account_id, start=start, end=end, limit=limit)
        return [dict(item) for item in self._history if (start is None or item["timestamp"] >= start)
                and (end is None or item["timestamp"] <= end)][-max(0, int(limit)):]

    def summary(self) -> Dict[str, Any]:
        returns = [item["return"] for item in self._returns]
        sharpe = None
        volatility = None
        if len(returns) >= 2:
            volatility = statistics.stdev(returns)
            if volatility > 0:
                sharpe = statistics.mean(returns) / volatility
                if self.periods_per_year is not None:
                    sharpe *= math.sqrt(self.periods_per_year)
                    volatility *= math.sqrt(self.periods_per_year)
        markouts = {}
        for horizon in (1, 5, 30):
            values = [item["signed_markout_bps"] for item in self._markouts
                      if item["horizon_seconds"] == horizon and item["signed_markout_bps"] is not None]
            markouts[str(horizon)] = {"count": len(values),
                                      "mean_bps": statistics.mean(values) if values else None}
        return {**self._latest_summary, "schema_version": 1,
                "return_interval_seconds": 60, "return_observations": len(returns),
                "sharpe_ratio": sharpe, "volatility": volatility,
                "annualized": self.periods_per_year is not None,
                "periods_per_year": self.periods_per_year,
                "max_drawdown": self._max_drawdown, "markouts": markouts,
                "metrics_basis": "cash_flow_adjusted_complete_minute_returns",
                "execution_mode": "local_paper"}
