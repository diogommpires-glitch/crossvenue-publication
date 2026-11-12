"""Source precedence and economic-time settlement regressions."""

import pytest

from src.runtime.portfolio_ledger import PortfolioLedger
from src.utils.types import Side
from tests.test_portfolio_ledger import make_equity, make_future

NOW = 1792929600.0


def open_ledger(spec):
    ledger = PortfolioLedger(10000, max_mark_age_seconds=30, clock=lambda: NOW + 1000)
    ledger.register_instrument(spec)
    ledger.record_fill(
        "opening", spec.instrument_id, Side.BUY, 1, 100, fee=0, fee_currency="USD", timestamp=NOW
    )
    return ledger


def official_close(ledger, price, stamp):
    ledger.set_valuation_mark(
        "AAPL",
        price,
        stamp,
        valid_until=NOW + 2000,
        source="sourced fixture",
        basis="official_close",
        stress_fraction=0.1,
        processing_timestamp=NOW + 10,
    )


def test_older_official_close_preserves_newer_observed_price_time_and_equity():
    ledger = open_ledger(make_equity())
    ledger.mark_instrument("AAPL", 110, NOW + 2)
    before = ledger.snapshot(NOW + 2)
    official_close(ledger, 150, NOW + 1)
    after = ledger.snapshot(NOW + 2)
    position = after.positions["AAPL"]
    assert (position.mark_price, position.mark_timestamp) == (110, NOW + 2)
    assert position.mark_basis == "observed_quote"
    assert after.equity == before.equity
    with pytest.raises(ValueError, match="stale mark"):
        ledger.snapshot(NOW + 1000)


def test_older_official_close_preserves_newer_official_eligibility_and_stress():
    ledger = open_ledger(make_equity())
    official_close(ledger, 110, NOW + 2)
    official_close(ledger, 150, NOW + 1)
    value = ledger.snapshot(NOW + 1000)
    position = value.positions["AAPL"]
    assert (position.mark_price, position.mark_timestamp) == (110, NOW + 2)
    assert position.mark_basis == "official_close"
    assert ledger.closed_market_stress(NOW + 1000) == pytest.approx(11)
    assert ledger.export_state()["valuation_marks"]["AAPL"]["price"] == 110


def test_equal_time_differing_observed_mark_cannot_inherit_official_eligibility():
    ledger = open_ledger(make_equity())
    official_close(ledger, 101, NOW)
    ledger.mark_instrument("AAPL", 102, NOW)
    position = ledger.get_position("AAPL", as_of=NOW)
    assert position.mark_basis == "observed_quote"
    assert position.valuation_valid_until is None
    assert ledger.closed_market_stress(NOW + 1000) == 0
    with pytest.raises(ValueError, match="stale mark"):
        ledger.snapshot(NOW + 1000)
    restored = ledger.clone()
    assert restored.get_position("AAPL", as_of=NOW).mark_basis == "observed_quote"
    with pytest.raises(ValueError, match="stale mark"):
        restored.snapshot(NOW + 1000)


def test_equal_time_official_close_can_supersede_observed_mark_without_rewinding():
    ledger = open_ledger(make_equity())
    official_close(ledger, 101, NOW)
    position = ledger.snapshot(NOW + 1000).positions["AAPL"]
    assert (position.mark_price, position.mark_timestamp, position.mark_basis) == (
        101,
        NOW,
        "official_close",
    )
    before = ledger.export_state()
    with pytest.raises(ValueError, match="backward"):
        ledger.mark_instrument("AAPL", 99, NOW - 1)
    assert ledger.export_state() == before


def test_sourced_settlement_preserves_newer_observed_mark_and_conserves_equity():
    ledger = open_ledger(make_future())
    ledger.mark_instrument("MESZ26", 110, NOW + 20)
    before = ledger.snapshot(NOW + 20)
    variation = ledger.record_sourced_settlement("settlement-1", "MESZ26", 105, NOW + 10)
    after = ledger.snapshot(NOW + 20)
    position = after.positions["MESZ26"]
    assert variation == pytest.approx(250)
    assert after.cash_by_currency["USD"] == pytest.approx(10250)
    assert position.average_entry_price == 105
    assert (position.mark_price, position.mark_timestamp, position.mark_basis) == (
        110,
        NOW + 20,
        "observed_quote",
    )
    assert (position.realized_pnl, position.unrealized_pnl) == pytest.approx((250, 250))
    assert after.equity == pytest.approx(before.equity)
    assert after.total_pnl == pytest.approx(before.total_pnl)
    with pytest.raises(ValueError, match="stale mark"):
        ledger.snapshot(NOW + 1000)


@pytest.mark.parametrize("effective_offset", [0, 10])
def test_sourced_settlement_uses_official_source_time_when_it_is_latest(effective_offset):
    ledger = open_ledger(make_future())
    stamp = NOW + effective_offset
    assert ledger.record_sourced_settlement("settlement-1", "MESZ26", 105, stamp) == pytest.approx(
        250
    )
    value = ledger.snapshot(stamp)
    position = value.positions["MESZ26"]
    assert (position.mark_price, position.mark_timestamp) == (105, stamp)
    assert position.mark_basis == "official_settlement"
    assert position.valuation_valid_until is None
    assert ledger.closed_market_stress(NOW + 1000) == 0
    assert value.equity == pytest.approx(10250)
    with pytest.raises(ValueError, match="stale mark"):
        ledger.snapshot(NOW + 1000)


def test_sourced_settlement_retry_and_conflict_survive_checkpoint_restore():
    ledger = open_ledger(make_future())
    assert ledger.record_sourced_settlement("settlement-1", "MESZ26", 105, NOW + 10) == 250
    restored = ledger.clone()
    before = restored.export_state()
    assert restored.record_sourced_settlement("settlement-1", "MESZ26", 105, NOW + 10) == 0
    assert restored.export_state() == before
    with pytest.raises(ValueError, match="reused"):
        restored.record_sourced_settlement("settlement-1", "MESZ26", 106, NOW + 10)
    assert restored.export_state() == before


def test_sourced_settlement_rejects_reversed_economic_order_after_restore():
    ledger = open_ledger(make_future())
    ledger.record_sourced_settlement("settlement-new", "MESZ26", 105, NOW + 20)
    restored = ledger.clone()
    before = restored.export_state()
    with pytest.raises(ValueError, match="settlement.*preced"):
        restored.record_sourced_settlement("settlement-old", "MESZ26", 104, NOW + 10)
    assert restored.export_state() == before


def test_direct_settlement_keeps_its_existing_monotonic_mark_guard():
    ledger = open_ledger(make_future())
    ledger.mark_instrument("MESZ26", 110, NOW + 20)
    before = ledger.export_state()
    with pytest.raises(ValueError, match="settlement timestamp precedes the latest mark"):
        ledger.settle_future("MESZ26", 105, NOW + 10)
    assert ledger.export_state() == before


@pytest.mark.parametrize(
    "action_id, price, stamp, message",
    [
        ("", 105, NOW + 1, "ID"),
        ("bad-price", 105.1, NOW + 1, "tick"),
        ("bad-time", 105, -1, "timestamp"),
    ],
)
def test_invalid_sourced_settlement_cannot_mutate_financial_state(action_id, price, stamp, message):
    ledger = open_ledger(make_future())
    before = ledger.export_state()
    with pytest.raises(ValueError, match=message):
        ledger.record_sourced_settlement(action_id, "MESZ26", price, stamp)
    assert ledger.export_state() == before


def test_settlement_cannot_use_positions_opened_after_effective_time_when_recent_log_evicted():
    from tests.test_execution_safety import NOW as ENGINE_NOW
    from tests.test_execution_safety import engine_for

    engine, clock = engine_for("CME:ESZ6")
    engine.ledger.record_fill(
        "later-open",
        "CME:ESZ6",
        Side.BUY,
        1,
        5000,
        fee=0,
        fee_currency="USD",
        timestamp=ENGINE_NOW + 20,
    )
    engine.trades = [{"instrument_id": "other-stream", "timestamp": ENGINE_NOW + 25}] * 2000
    clock.value = ENGINE_NOW + 30
    engine.schedule_account_action(
        {
            "action_id": "before-opening",
            "action_type": "settlement",
            "instrument_id": "CME:ESZ6",
            "source": "sourced historical settlement",
            "effective_ts": ENGINE_NOW + 10,
            "payload": {"settlement_price": 4990},
        }
    )
    before = engine.ledger.export_state()
    action = engine.process_account_actions(clock.value)[0]
    assert action["status"] == "failed"
    assert "fill" in action["error"]
    assert engine.ledger.export_state() == before
    assert engine.ledger.cash_by_currency["USD"] == 40000


def test_fill_watermark_is_the_maximum_and_survives_checkpoint_restore():
    ledger = open_ledger(make_future())
    for fill_id, stamp in (("later-fill", NOW + 20), ("delayed-earlier-fill", NOW + 10)):
        ledger.record_fill(
            fill_id, "MESZ26", Side.BUY, 1, 110, fee=0, fee_currency="USD", timestamp=stamp
        )
    ledger.mark_instrument("MESZ26", 120, NOW + 30)
    restored = ledger.clone()
    position = restored.get_position("MESZ26", as_of=NOW + 30)
    assert position.last_fill_timestamp == NOW + 20
    assert position.fill_history_verified
    before = restored.export_state()
    with pytest.raises(ValueError, match="settlement.*fill"):
        restored.record_sourced_settlement("before-fill", "MESZ26", 105, NOW + 15)
    assert restored.export_state() == before
    before_equity = restored.snapshot(NOW + 30).equity
    assert restored.record_sourced_settlement("at-fill", "MESZ26", 110, NOW + 20) == pytest.approx(
        500
    )
    assert restored.snapshot(NOW + 30).equity == pytest.approx(before_equity)


def test_late_fill_cannot_rewrite_committed_settlement_cash_or_basis_after_restore():
    ledger = open_ledger(make_future())
    ledger.record_sourced_settlement("settlement", "MESZ26", 105, NOW + 10)
    restored = ledger.clone()
    before = restored.export_state()
    with pytest.raises(ValueError, match="fill.*settlement"):
        restored.record_fill(
            "late-opening", "MESZ26", Side.BUY, 1, 90, fee=0, fee_currency="USD", timestamp=NOW + 5
        )
    assert restored.export_state() == before
    value = restored.snapshot(NOW + 10)
    assert value.cash_by_currency["USD"] == 10250
    assert value.positions["MESZ26"].average_entry_price == 105
    assert value.positions["MESZ26"].quantity == 1
    assert value.equity == 10250
    assert not restored.record_fill(
        "opening", "MESZ26", Side.BUY, 1, 100, fee=0, fee_currency="USD", timestamp=NOW
    )
    assert restored.export_state() == before


def test_legacy_missing_fill_history_remains_unverified_after_new_fill_and_restore():
    ledger = open_ledger(make_future())
    checkpoint = ledger.export_state()
    checkpoint["positions"]["MESZ26"].pop("last_fill_timestamp", None)
    checkpoint["positions"]["MESZ26"].pop("fill_history_verified", None)
    ledger.restore_state(checkpoint)
    position = ledger.get_position("MESZ26", as_of=NOW)
    assert position.last_fill_timestamp is None
    assert not position.fill_history_verified
    ledger.record_fill(
        "known-new-fill", "MESZ26", Side.BUY, 1, 110, fee=0, fee_currency="USD", timestamp=NOW + 20
    )
    restored = ledger.clone()
    assert not restored.get_position("MESZ26", as_of=NOW + 20).fill_history_verified
    before = restored.export_state()
    with pytest.raises(ValueError, match="verified fill history"):
        restored.record_sourced_settlement("backdated", "MESZ26", 105, NOW + 30)
    assert restored.export_state() == before


def test_legacy_current_settlement_does_not_invent_historical_fill_timestamp():
    ledger = open_ledger(make_future())
    checkpoint = ledger.export_state()
    checkpoint["positions"]["MESZ26"].pop("last_fill_timestamp", None)
    checkpoint["positions"]["MESZ26"].pop("fill_history_verified", None)
    ledger.restore_state(checkpoint)
    assert ledger.record_sourced_settlement("current", "MESZ26", 105, NOW + 1000) == 250
    position = ledger.get_position("MESZ26", as_of=NOW + 1000)
    assert position.last_fill_timestamp is None
    assert not position.fill_history_verified
    assert ledger.snapshot(NOW + 1000).equity == 10250


def test_sourced_settlement_provenance_survives_restore_but_not_differing_observed_price():
    ledger = open_ledger(make_future())
    ledger.record_sourced_settlement("official", "MESZ26", 105, NOW + 10)
    restored = ledger.clone()
    assert restored.get_position("MESZ26", as_of=NOW + 10).mark_basis == "official_settlement"
    restored.mark_instrument("MESZ26", 106, NOW + 10)
    restored = restored.clone()
    position = restored.get_position("MESZ26", as_of=NOW + 10)
    assert position.mark_basis == "observed_quote"
    assert position.valuation_valid_until is None
    assert restored.closed_market_stress(NOW + 1000) == 0
    assert restored.record_sourced_settlement("official", "MESZ26", 105, NOW + 10) == 0
    assert restored.get_position("MESZ26", as_of=NOW + 10).mark_basis == "observed_quote"
    with pytest.raises(ValueError, match="stale mark"):
        restored.snapshot(NOW + 1000)


def test_direct_settlement_cannot_inherit_an_earlier_sourced_settlement_label():
    ledger = open_ledger(make_future())
    ledger.record_sourced_settlement("official", "MESZ26", 105, NOW + 10)
    assert ledger.get_position("MESZ26", as_of=NOW + 10).mark_basis == "official_settlement"
    ledger.settle_future("MESZ26", 105, NOW + 20)
    assert ledger.clone().get_position("MESZ26", as_of=NOW + 20).mark_basis == "observed_quote"


def test_legacy_settlement_timestamp_does_not_invent_a_sourced_mark_label():
    ledger = open_ledger(make_future())
    ledger.record_sourced_settlement("official", "MESZ26", 105, NOW + 10)
    checkpoint = ledger.export_state()
    checkpoint["positions"]["MESZ26"].pop("sourced_settlement_price", None)
    ledger.restore_state(checkpoint)
    assert ledger.get_position("MESZ26", as_of=NOW + 10).mark_basis == "observed_quote"


@pytest.mark.parametrize("price, settlement_time", [(105.1, NOW + 10), (105, None)])
def test_checkpoint_rejects_malformed_sourced_settlement_provenance(price, settlement_time):
    ledger = open_ledger(make_future())
    checkpoint = ledger.export_state()
    checkpoint["positions"]["MESZ26"].update(
        sourced_settlement_price=price,
        settlement_timestamp=settlement_time,
        mark_timestamp=NOW + 10,
    )
    before = ledger.export_state()
    with pytest.raises(ValueError, match="sourced settlement"):
        ledger.restore_state(checkpoint)
    assert ledger.export_state() == before
