import pytest

from src.runtime.risk_manager import RiskLimits, RiskManager, RiskViolation


def manager(**overrides):
    values = dict(max_position_size=1.0, max_order_size=0.5, max_daily_loss=100.0,
                  max_total_exposure=20_000.0, max_orders_per_minute=2,
                  circuit_breaker_loss_pct=0.1, min_balance_reserve=50.0)
    values.update(overrides)
    risk = RiskManager(RiskLimits(**values))
    risk.estimated_fee_rate = 0.01
    risk.reset(balance=1_000.0)
    risk.update_state(0.0, 1_000.0, 1_000.0, 1_000.0)
    return risk


def test_pending_both_sides_count_toward_worst_case_position():
    risk = manager(max_position_size=0.7)
    risk.allow_short = True
    assert risk.reserve_order("bid", "buy", 0.4, 100.0)[0]
    assert not risk.reserve_order("bid2", "buy", 0.4, 100.0)[0]
    assert risk.reserve_order("ask", "sell", 0.4, 100.0)[0]
    assert risk.get_status()["worst_case_position"] == pytest.approx(0.4)


def test_reserved_buy_includes_fee_estimate_and_cash_reserve():
    risk = manager(min_balance_reserve=50.0)
    risk.update_state(0.0, 1_000.0, 1_000.0, 99.0)
    assert risk.reserve_order("ok", "buy", 0.49, 100.0)[0] is False
    assert risk.reserve_order("small", "buy", 0.4, 100.0)[0] is True


def test_partial_fill_and_cancel_release_pending_reservations():
    risk = manager()
    assert risk.reserve_order("bid", "buy", 0.4, 100.0)[0]
    risk.update_pending_order("bid", 0.1)
    assert risk.get_status()["pending_orders_count"] == 1
    risk.clear_pending_order("bid")
    assert risk.get_status()["pending_orders_count"] == 0


def test_equity_loss_trips_circuit_breaker_and_blocks_new_orders():
    risk = manager(circuit_breaker_loss_pct=0.05)
    risk.update_state(0.0, 949.0, 1_000.0, 949.0)
    valid, violation, _ = risk.validate_order("buy", 0.1, 100.0)
    assert not valid
    assert violation == RiskViolation.CIRCUIT_BREAKER


def test_concurrent_reservations_cannot_overrun_position_limit():
    from concurrent.futures import ThreadPoolExecutor

    risk = manager(max_position_size=0.5, max_orders_per_minute=100)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda index: risk.reserve_order(str(index), "buy", 0.2, 100.0), range(8)
        ))
    accepted = sum(1 for result in results if result[0])
    assert accepted == 2
    assert sum(float(order["quantity"]) for order in risk.pending_orders.values()) <= 0.5


def test_spot_paper_account_rejects_unfunded_sells_and_reserves_inventory():
    risk = manager(max_position_size=2.0)
    valid, violation, _ = risk.reserve_order("short", "sell", 0.1, 100.0)
    assert not valid
    assert violation == RiskViolation.INSUFFICIENT_INVENTORY

    risk.update_state(0.5, 1_000.0, 1_000.0, 1_000.0)
    assert risk.reserve_order("ask", "sell", 0.4, 100.0)[0]
    valid, violation, _ = risk.reserve_order("too-large-ask", "sell", 0.2, 100.0)
    assert not valid
    assert violation == RiskViolation.INSUFFICIENT_INVENTORY


def test_duplicate_pending_order_id_cannot_overwrite_reservation():
    risk = manager()
    assert risk.reserve_order("same", "buy", 0.1, 100.0)[0]
    accepted, _, message = risk.reserve_order("same", "buy", 0.2, 100.0)
    assert not accepted
    assert "already" in message
    assert risk.pending_orders["same"]["quantity"] == 0.1
