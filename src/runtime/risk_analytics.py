"""Rolling account-risk analytics based on mark-to-market equity.

P&L observations are cumulative quote-currency P&L. If `equity` is omitted,
equity is derived as starting_equity + pnl. Returns are per observation unless
the caller explicitly supplies periods_per_year and requests annualization.
VaR and CVaR are signed quote-currency values (losses are negative).
"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RiskMetrics:
    """Risk snapshot; VaR/CVaR/drawdown are quote currency."""

    var_95: float
    cvar_95: float
    volatility: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    position_risk: float
    position_notional: Optional[float] = None


@dataclass
class StressTestResult:
    """Estimated scenario result; all P&L and impact values use quote currency."""

    scenario_name: str
    estimated_loss: float
    estimated_pnl: float
    position_impact: float
    description: str


class RiskAnalytics:
    """Compute rolling risk from equity changes and explicit market scenarios."""

    def __init__(
        self,
        window_size: int = 100,
        confidence_level: float = 0.95,
        risk_free_rate: float = 0.0,
        starting_equity: float = 10000.0,
        periods_per_year: Optional[float] = None,
    ) -> None:
        if not isinstance(window_size, int) or window_size < 2:
            raise ValueError("window_size must be an integer of at least 2")
        if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if not math.isfinite(risk_free_rate):
            raise ValueError("risk_free_rate must be finite")
        if not math.isfinite(starting_equity) or starting_equity <= 0:
            raise ValueError("starting_equity must be finite and positive")
        if periods_per_year is not None and (not math.isfinite(periods_per_year) or periods_per_year <= 0):
            raise ValueError("periods_per_year must be finite and positive")
        if risk_free_rate != 0 and periods_per_year is None:
            raise ValueError("periods_per_year is required when risk_free_rate is non-zero")

        self.window_size = window_size
        self.confidence_level = float(confidence_level)
        self.risk_free_rate = float(risk_free_rate)
        self.starting_equity = float(starting_equity)
        self.periods_per_year = periods_per_year
        self.pnl_history: Deque[float] = deque(maxlen=window_size)
        self.equity_history: Deque[float] = deque(maxlen=window_size)
        self.return_history: Deque[float] = deque(maxlen=window_size)
        self.position_history: Deque[float] = deque(maxlen=window_size)
        self.timestamp_history: Deque[float] = deque(maxlen=window_size)
        self.current_pnl = 0.0
        self.current_equity = self.starting_equity
        self.current_price: Optional[float] = None
        self.peak_pnl = 0.0
        self.peak_equity = self.starting_equity
        self.max_drawdown = 0.0
        self._last_equity: Optional[float] = None
        self._last_timestamp: Optional[float] = None

    def update(
        self,
        pnl: float,
        position: float,
        timestamp: Optional[float] = None,
        equity: Optional[float] = None,
        current_price: Optional[float] = None,
    ) -> None:
        """Add a cumulative P&L/equity observation and signed position.

        P&L deltas are never divided by prior P&L. Returns use account equity;
        equity must remain positive for percentage returns to be meaningful.
        """
        if not math.isfinite(pnl) or not math.isfinite(position):
            raise ValueError("pnl and position must be finite")
        observed_equity = self.starting_equity + pnl if equity is None else float(equity)
        if not math.isfinite(observed_equity) or observed_equity <= 0:
            raise ValueError("equity must be finite and positive")
        expected_equity = self.starting_equity + pnl
        if not math.isclose(observed_equity, expected_equity, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError("equity must equal starting_equity plus cumulative pnl")
        if timestamp is not None:
            timestamp = float(timestamp)
            if not math.isfinite(timestamp):
                raise ValueError("timestamp must be finite")
            if self._last_timestamp is not None and timestamp < self._last_timestamp:
                raise ValueError("timestamps must be non-decreasing")
        if current_price is not None and (not math.isfinite(current_price) or current_price <= 0):
            raise ValueError("current_price must be finite and positive")

        if self._last_equity is not None:
            self.return_history.append((observed_equity - self._last_equity) / self._last_equity)
        self._last_equity = observed_equity
        self.current_equity = observed_equity
        self.current_pnl = observed_equity - self.starting_equity
        self.pnl_history.append(self.current_pnl)
        self.equity_history.append(observed_equity)
        self.position_history.append(float(position))
        self.peak_pnl = max(self.peak_pnl, self.current_pnl)
        self.peak_equity = max(self.peak_equity, observed_equity)
        self.max_drawdown = min(self.max_drawdown, observed_equity - self.peak_equity)
        if current_price is not None:
            self.current_price = float(current_price)
        if timestamp is not None:
            self.timestamp_history.append(timestamp)
            self._last_timestamp = timestamp

    def calculate_var(self, confidence_level: Optional[float] = None) -> float:
        """Historical VaR as a signed quote-currency loss at the chosen tail."""
        conf = self._validated_confidence(confidence_level)
        if len(self.return_history) < 10:
            return 0.0
        tail_return = float(np.percentile(np.asarray(self.return_history), (1.0 - conf) * 100.0))
        return min(0.0, tail_return) * self.current_equity

    def calculate_cvar(self, confidence_level: Optional[float] = None) -> float:
        """Historical expected shortfall, signed and expressed in quote currency."""
        conf = self._validated_confidence(confidence_level)
        if len(self.return_history) < 10:
            return 0.0
        returns = np.asarray(self.return_history, dtype=float)
        threshold = float(np.percentile(returns, (1.0 - conf) * 100.0))
        tail = returns[returns <= threshold]
        return min(0.0, float(np.mean(tail))) * self.current_equity if tail.size else 0.0

    def calculate_volatility(self, annualize: bool = False) -> float:
        """Return sample equity-return volatility, optionally annualized by explicit cadence."""
        scale = self._annualization_scale(annualize)
        if len(self.return_history) < 2:
            return 0.0
        return float(np.std(np.asarray(self.return_history), ddof=1)) * scale


    def calculate_drawdown(self) -> Tuple[float, float]:
        """Return maximum and current peak-to-trough drawdown in quote currency."""
        return float(self.max_drawdown), float(self.current_equity - self.peak_equity)

    def calculate_sharpe_ratio(self, annualize: bool = False) -> float:
        """Return the Sharpe ratio per observation, or annualize with explicit cadence."""
        scale = self._annualization_scale(annualize)
        if self.risk_free_rate and self.periods_per_year is None:
            raise ValueError("periods_per_year is required to apply annual risk-free rate")
        if len(self.return_history) < 2:
            return 0.0
        returns = np.asarray(self.return_history, dtype=float)
        per_period_rf = self.risk_free_rate / self.periods_per_year if self.periods_per_year else 0.0
        excess = returns - per_period_rf
        deviation = float(np.std(excess, ddof=1))
        if deviation <= 1e-15:
            mean_excess = float(np.mean(excess))
            ratio = math.inf if mean_excess > 0 else (-math.inf if mean_excess < 0 else 0.0)
        else:
            ratio = float(np.mean(excess) / deviation)
        return ratio * scale


    def get_risk_metrics(self, current_price: Optional[float] = None) -> RiskMetrics:
        """Return one internally consistent risk snapshot."""
        price = self.current_price if current_price is None else current_price
        if price is not None and (not math.isfinite(price) or price <= 0):
            raise ValueError("current_price must be finite and positive")
        max_dd, current_dd = self.calculate_drawdown()
        position = float(self.position_history[-1]) if self.position_history else 0.0
        return RiskMetrics(
            var_95=self.calculate_var(),
            cvar_95=self.calculate_cvar(),
            volatility=self.calculate_volatility(),
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            sharpe_ratio=self.calculate_sharpe_ratio(),
            position_risk=abs(position),
            position_notional=abs(position) * price if price is not None else None,
        )

    def generate_position_heatmap(
        self, current_price: float, current_position: float, price_range_pct: float = 0.10
    ) -> Dict[float, float]:
        """Map hypothetical prices to incremental mark-to-market P&L."""
        self._validate_price_position(current_price, current_position)
        if not math.isfinite(price_range_pct) or not 0 <= price_range_pct < 1:
            raise ValueError("price_range_pct must be in [0, 1)")
        if current_position == 0:
            return {}
        prices = np.linspace(current_price * (1 - price_range_pct), current_price * (1 + price_range_pct), 21)
        return {float(price): float(current_position * (price - current_price)) for price in prices}

    def stress_test_volatility_spike(
        self, current_price: float, current_position: float, vol_multiplier: float = 2.0
    ) -> StressTestResult:
        """Estimate a two-observation-standard-deviation adverse move."""
        self._validate_price_position(current_price, current_position)
        if not math.isfinite(vol_multiplier) or vol_multiplier < 0:
            raise ValueError("vol_multiplier must be finite and non-negative")
        current_vol = self.calculate_volatility(annualize=False)
        adverse_move_pct = min(0.99, 2.0 * current_vol * vol_multiplier)
        if current_position > 0:
            change = -current_price * adverse_move_pct
        elif current_position < 0:
            change = current_price * adverse_move_pct
        else:
            change = 0.0
        loss = current_position * change
        return StressTestResult(
            scenario_name="Volatility Spike",
            estimated_loss=loss,
            estimated_pnl=self.current_pnl + loss,
            position_impact=abs(loss),
            description=f"Adverse move of {2.0 * vol_multiplier:.2f} observation standard deviations",
        )

    def stress_test_liquidity_shock(
        self, current_price: float, current_position: float, slippage_bps: float = 100.0
    ) -> StressTestResult:
        """Estimate forced liquidation cost at an explicit slippage level."""
        self._validate_price_position(current_price, current_position)
        if not math.isfinite(slippage_bps) or slippage_bps < 0:
            raise ValueError("slippage_bps must be finite and non-negative")
        loss = -abs(current_position) * current_price * slippage_bps / 10000.0
        return StressTestResult(
            scenario_name="Liquidity Shock",
            estimated_loss=loss,
            estimated_pnl=self.current_pnl + loss,
            position_impact=abs(loss),
            description=f"Forced liquidation with {slippage_bps:.0f} bps slippage",
        )

    def stress_test_flash_crash(
        self, current_price: float, current_position: float, crash_pct: float = 0.10
    ) -> StressTestResult:
        """Estimate an adverse price shock for the current signed position."""
        self._validate_price_position(current_price, current_position)
        if not math.isfinite(crash_pct) or not 0 <= crash_pct < 1:
            raise ValueError("crash_pct must be in [0, 1)")
        move = -current_price * crash_pct if current_position >= 0 else current_price * crash_pct
        loss = current_position * move
        direction = "drop" if current_position >= 0 else "spike"
        return StressTestResult(
            scenario_name="Flash Crash",
            estimated_loss=loss,
            estimated_pnl=self.current_pnl + loss,
            position_impact=abs(loss),
            description=f"Sudden {crash_pct * 100:.0f}% price {direction}",
        )

    def run_all_stress_tests(self, current_price: float, current_position: float) -> List[StressTestResult]:
        return [
            self.stress_test_volatility_spike(current_price, current_position),
            self.stress_test_liquidity_shock(current_price, current_position),
            self.stress_test_flash_crash(current_price, current_position),
        ]

    def reset(self) -> None:
        self.pnl_history.clear()
        self.equity_history.clear()
        self.return_history.clear()
        self.position_history.clear()
        self.timestamp_history.clear()
        self.current_pnl = 0.0
        self.current_equity = self.starting_equity
        self.current_price = None
        self.peak_pnl = 0.0
        self.peak_equity = self.starting_equity
        self.max_drawdown = 0.0
        self._last_equity = None
        self._last_timestamp = None

    def _validated_confidence(self, confidence_level: Optional[float]) -> float:
        confidence = self.confidence_level if confidence_level is None else confidence_level
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        return float(confidence)

    def _annualization_scale(self, annualize: bool) -> float:
        if not annualize:
            return 1.0
        if self.periods_per_year is None:
            raise ValueError("set periods_per_year before requesting annualized metrics")
        return math.sqrt(self.periods_per_year)

    @staticmethod
    def _validate_price_position(current_price: float, current_position: float) -> None:
        if not math.isfinite(current_price) or current_price <= 0:
            raise ValueError("current_price must be finite and positive")
        if not math.isfinite(current_position):
            raise ValueError("current_position must be finite")
