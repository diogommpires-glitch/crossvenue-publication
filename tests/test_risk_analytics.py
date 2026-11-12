import math

import numpy as np
import pytest

from src.runtime.risk_analytics import RiskAnalytics


def test_returns_use_equity_and_handle_near_zero_pnl():
    risk = RiskAnalytics(starting_equity=1_000.0, window_size=20)
    risk.update(pnl=0.001, equity=1_000.001, position=0.0)
    risk.update(pnl=10.001, equity=1_010.001, position=0.0)
    assert risk.return_history[-1] == pytest.approx(0.01)
    assert math.isfinite(risk.calculate_volatility())


def test_var_scales_with_equity_not_pnl_level():
    risk = RiskAnalytics(starting_equity=10_000.0, window_size=100)
    returns = [-0.02, -0.015, -0.01, -0.005] + [0.001] * 20
    equity = 10_000.0
    risk.update(pnl=0.0, equity=equity, position=0.0)
    for index, ret in enumerate(returns, 1):
        equity *= 1 + ret
        risk.update(pnl=equity - 10_000.0, equity=equity, position=0.0)
    assert risk.calculate_var() < 0
    assert risk.calculate_cvar() <= risk.calculate_var()
    assert abs(risk.calculate_var()) < equity * 0.1


def test_volatility_matches_equity_return_stream():
    risk = RiskAnalytics(window_size=100, starting_equity=1_000.0)
    rng = np.random.default_rng(42)
    equity = 1_000.0
    for ret in rng.normal(0.0, 0.02, 100):
        equity *= 1 + ret
        risk.update(pnl=equity - 1_000.0, equity=equity, position=0.0)
    assert risk.calculate_volatility() == pytest.approx(np.std(risk.return_history, ddof=1))
    assert risk.calculate_volatility() == pytest.approx(0.018, abs=0.004)


def test_drawdown_uses_starting_equity_baseline():
    risk = RiskAnalytics(starting_equity=1_000.0)
    for pnl in (100.0, 120.0, 140.0, 130.0, 90.0, 120.0):
        risk.update(pnl=pnl, equity=1_000.0 + pnl, position=0.0)
    max_dd, current_dd = risk.calculate_drawdown()
    assert max_dd == pytest.approx(-50.0)
    assert current_dd == pytest.approx(-20.0)


def test_max_drawdown_survives_window_eviction():
    risk = RiskAnalytics(starting_equity=100.0, window_size=2)
    for equity in (200.0, 150.0, 160.0, 170.0):
        risk.update(pnl=equity - 100.0, equity=equity, position=0.0)
    maximum, current = risk.calculate_drawdown()
    assert maximum == pytest.approx(-50.0)
    assert current == pytest.approx(-30.0)


def test_annualization_requires_explicit_periods():
    risk = RiskAnalytics(starting_equity=1_000.0)
    with pytest.raises(ValueError, match="periods_per_year"):
        risk.calculate_volatility(annualize=True)
    risk = RiskAnalytics(starting_equity=1_000.0, periods_per_year=252)
    for equity in (1_010.0, 1_012.0, 1_008.0, 1_015.0):
        risk.update(pnl=equity - 1_000.0, equity=equity, position=0.1)
    assert risk.calculate_volatility(annualize=True) == pytest.approx(risk.calculate_volatility() * np.sqrt(252))


def test_stress_outputs_are_quote_currency_and_directional():
    risk = RiskAnalytics(starting_equity=10_000.0)
    risk.update(pnl=0.0, equity=10_000.0, position=0.0)
    long = risk.stress_test_flash_crash(50_000.0, 0.5, 0.1)
    short = risk.stress_test_flash_crash(50_000.0, -0.5, 0.1)
    assert long.estimated_loss == pytest.approx(-2_500.0)
    assert short.estimated_loss == pytest.approx(-2_500.0)
    liquidity = risk.stress_test_liquidity_shock(50_000.0, 0.5, 100.0)
    assert liquidity.position_impact == pytest.approx(250.0)


def test_heatmap_is_incremental_pnl_and_reset_clears_history():
    risk = RiskAnalytics()
    heatmap = risk.generate_position_heatmap(100.0, 2.0, 0.1)
    assert min(heatmap.values()) == pytest.approx(-20.0)
    assert max(heatmap.values()) == pytest.approx(20.0)
    risk.update(pnl=10.0, equity=10_010.0, position=1.0, timestamp=1.0)
    risk.reset()
    assert len(risk.return_history) == 0
    assert risk.current_equity == 10_000.0


def test_invalid_equity_and_time_are_rejected():
    risk = RiskAnalytics()
    with pytest.raises(ValueError, match="equity"):
        risk.update(pnl=-10_000.0, equity=0.0, position=0.0)
    risk.update(pnl=0.0, equity=10_000.0, position=0.0, timestamp=2.0)
    with pytest.raises(ValueError, match="non-decreasing"):
        risk.update(pnl=1.0, equity=10_001.0, position=0.0, timestamp=1.0)


def test_equity_and_cumulative_pnl_must_reconcile():
    risk = RiskAnalytics(starting_equity=1_000.0)
    with pytest.raises(ValueError, match="starting_equity plus cumulative pnl"):
        risk.update(pnl=10.0, equity=1_011.0, position=0.0)


@pytest.mark.parametrize("method", ["calculate_var", "calculate_cvar"])
def test_tail_confidence_is_validated_even_before_minimum_sample(method):
    risk = RiskAnalytics()
    with pytest.raises(ValueError, match="confidence_level"):
        getattr(risk, method)(1.0)
