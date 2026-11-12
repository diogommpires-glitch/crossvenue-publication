"""Analytical funding and probability bounds, not simulated market evidence."""

import importlib
import importlib.util
import math

import pytest


def api():
    assert importlib.util.find_spec("src.runtime.feasibility_envelope") is not None, (
        "The approved analytical envelope is not implemented"
    )
    return importlib.import_module("src.runtime.feasibility_envelope")


def test_cash_bound_reserves_entry_fees_and_never_reuses_sale_proceeds():
    model = api()
    assert model.cash_round_trip_limit(10000, 1000, 1) == 9
    assert model.cash_round_trip_limit(10000, 1000, 1, settled_fraction=0.5) == 4
    assert model.cash_round_trip_limit(10000, 1000, 1, settled_fraction=0) == 0


def test_duration_bound_uses_entry_cutoff_and_flat_deadline():
    model = api()
    assert model.cycles_per_slot(1) == 375
    assert model.cycles_per_slot(60) == 6
    assert model.cycles_per_slot(60, entry_minutes=195, flat_minutes=205) == 3
    with pytest.raises(ValueError):
        model.cycles_per_slot(0)


def test_required_edge_matches_independent_nine_trade_gaussian_example():
    model = api()
    result = model.required_net_edge(0.55, 30, 9, correlation=0.1)
    assert result["all_in_net_bps"] == pytest.approx(1.685924, abs=0.000001)
    assert result["effective_count"] == pytest.approx(5)
    assert model.required_net_edge(0.60, 30, 9, correlation=0.1)["all_in_net_bps"] == pytest.approx(
        3.399008, abs=0.000001
    )
    higher = model.required_net_edge(0.55, 30, 9, correlation=1)
    assert higher["effective_count"] == 1
    assert higher["all_in_net_bps"] > result["all_in_net_bps"]


def test_all_session_green_target_accounts_for_inactive_days():
    model = api()
    assert model.required_net_edge(0.60, 30, 10, activity=0.5) is None
    active = model.required_net_edge(0.60, 30, 10, activity=0.75)
    assert active["traded_day_green_target"] == pytest.approx(0.8)
    assert active["all_in_net_bps"] > model.required_net_edge(0.60, 30, 10)["all_in_net_bps"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_financial_inputs_fail_explicitly(bad):
    model = api()
    with pytest.raises(ValueError):
        model.cash_round_trip_limit(bad, 1000, 1)
    with pytest.raises(ValueError):
        model.required_net_edge(0.55, bad, 9)


def test_fixed_cost_allocation_uses_actual_session_count():
    model = api()
    assert model.session_overhead(199, 20) == pytest.approx(9.95)
    assert model.session_overhead(199, 22) == pytest.approx(199 / 22)
    with pytest.raises(ValueError):
        model.session_overhead(199, 0)
    assert math.isfinite(model.session_overhead(398, 21))
