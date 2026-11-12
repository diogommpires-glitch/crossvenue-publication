from datetime import datetime, timezone
import time

import pytest

from src.market.instruments import AssetClass, FXAccountingMode, InstrumentSpec
from src.runtime.portfolio_ledger import PortfolioLedger
from src.utils.types import Side


def make_equity(*, instrument_id="AAPL", quote="USD", borrow_limit=None, borrow_fee_rate=None,
                contract_multiplier=1.0):
    return InstrumentSpec(
        instrument_id=instrument_id,
        asset_class=AssetClass.EQUITY,
        symbol=instrument_id,
        venue="NASDAQ",
        base_currency=instrument_id,
        quote_currency=quote,
        settlement_currency=quote,
        quantity_unit="shares",
        tick_size=0.01,
        quantity_step=1,
        min_quantity=1,
        contract_multiplier=contract_multiplier,
        borrow_limit=borrow_limit,
        borrow_fee_rate=borrow_fee_rate,
    )


def make_fx(*, mode=FXAccountingMode.DELIVERABLE, instrument_id="EURUSD",
            initial_margin_rate=None, maintenance_margin_rate=None,
            contract_multiplier=1.0, borrow_limit=None, borrow_fee_rate=None):
    return InstrumentSpec(
        instrument_id=instrument_id,
        asset_class=AssetClass.FX,
        symbol=instrument_id,
        venue="OANDA-DATA",
        base_currency="EUR",
        quote_currency="USD",
        settlement_currency="USD",
        quantity_unit="EUR",
        tick_size=0.0001,
        quantity_step=1000,
        min_quantity=1000,
        contract_multiplier=contract_multiplier,
        borrow_limit=borrow_limit,
        borrow_fee_rate=borrow_fee_rate,
        fx_accounting_mode=mode,
        initial_margin_rate=initial_margin_rate,
        maintenance_margin_rate=maintenance_margin_rate,
    )


def make_future(*, initial_margin_per_contract=500, maintenance_margin_per_contract=375):
    return InstrumentSpec(
        instrument_id="MESZ26",
        asset_class=AssetClass.FUTURE,
        symbol="MESZ26",
        venue="CME",
        base_currency="MES",
        quote_currency="USD",
        settlement_currency="USD",
        quantity_unit="contracts",
        tick_size=0.25,
        quantity_step=1,
        min_quantity=1,
        contract_multiplier=50,
        expiry=datetime(2026, 12, 18, tzinfo=timezone.utc),
        initial_margin_per_contract=initial_margin_per_contract,
        maintenance_margin_per_contract=maintenance_margin_per_contract,
    )


def fill(ledger, fill_id, instrument_id, side, quantity, price, *, fee=0.0,
         fee_currency="USD", timestamp=None, fill_instrument_id=None):
    return ledger.record_fill(
        fill_id, instrument_id, side, quantity, price,
        fee=fee, fee_currency=fee_currency,
        timestamp=time.time() if timestamp is None else timestamp,
        fill_instrument_id=fill_instrument_id,
    )


def test_equity_fills_reconcile_cash_position_fees_and_realized_pnl():
    now = time.time()
    ledger = PortfolioLedger(10_000, max_mark_age_seconds=60)
    ledger.register_instrument(make_equity())

    assert fill(ledger, "buy-1", "AAPL", Side.BUY, 10, 100, fee=1, timestamp=now)
    ledger.mark_instrument("AAPL", 110, now + 1)
    first = ledger.snapshot(now + 1)
    assert first.cash_by_currency["USD"] == pytest.approx(8_999)
    assert first.positions["AAPL"].quantity == pytest.approx(10)
    assert first.equity == pytest.approx(10_099)
    assert first.unrealized_pnl == pytest.approx(100)
    assert first.total_pnl == pytest.approx(99)
    assert first.fees_paid == pytest.approx(1)

    fill(ledger, "sell-1", "AAPL", Side.SELL, 4, 120, fee=0.5, timestamp=now + 2)
    ledger.mark_instrument("AAPL", 120, now + 3)
    fill(ledger, "sell-2", "AAPL", Side.SELL, 6, 120, fee=0.75, timestamp=now + 4)
    closed = ledger.snapshot(now + 4)
    assert closed.positions["AAPL"].quantity == 0
    assert closed.positions["AAPL"].realized_pnl == pytest.approx(200)
    assert closed.realized_pnl == pytest.approx(200)
    assert closed.equity == pytest.approx(10_197.75)
    assert closed.total_pnl == pytest.approx(197.75)
    assert closed.fees_paid == pytest.approx(2.25)
    assert closed.total_pnl == pytest.approx(
        closed.realized_pnl + closed.unrealized_pnl - closed.fees_paid
        + closed.translation_pnl
    )


def test_fill_ids_are_idempotent_and_conflicting_reuse_fails():
    now = time.time()
    ledger = PortfolioLedger(1_000)
    ledger.register_instrument(make_equity())
    args = ("fill-1", "AAPL", "buy", 1, 100)
    assert fill(ledger, *args, timestamp=now)
    assert not fill(ledger, *args, timestamp=now)
    assert ledger.snapshot(now).positions["AAPL"].quantity == 1
    with pytest.raises(ValueError, match="different fill data"):
        fill(ledger, "fill-1", "AAPL", "buy", 2, 100, timestamp=now)
    with pytest.raises(ValueError, match="does not match"):
        fill(ledger, "fill-2", "AAPL", "buy", 1, 100,
             timestamp=now, fill_instrument_id="MSFT")


def test_deliverable_fx_moves_both_currency_balances_and_marks_once():
    now = time.time()
    ledger = PortfolioLedger(10_000, max_mark_age_seconds=60)
    ledger.register_instrument(make_fx())
    fill(ledger, "fx-buy", "EURUSD", Side.BUY, 1_000, 1.1, fee=1, timestamp=now)
    ledger.set_fx_mark("EUR", 1.1, now)
    ledger.mark_instrument("EURUSD", 1.1, now)
    opened = ledger.snapshot(now)
    assert opened.cash_by_currency["EUR"] == pytest.approx(1_000)
    assert opened.cash_by_currency["USD"] == pytest.approx(8_899)
    assert opened.equity == pytest.approx(9_999)
    assert opened.unrealized_pnl == pytest.approx(0)

    ledger.mark_instrument("EURUSD", 1.2, now + 1)
    ledger.set_fx_mark("EUR", 1.2, now + 1)
    moved = ledger.snapshot(now + 1)
    assert moved.equity == pytest.approx(10_099)
    assert moved.unrealized_pnl == pytest.approx(100)
    assert moved.total_pnl == pytest.approx(moved.realized_pnl + moved.unrealized_pnl
                                            - moved.fees_paid + moved.translation_pnl)

    fill(ledger, "fx-sell", "EURUSD", Side.SELL, 1_000, 1.2, fee=1, timestamp=now + 2)
    ledger.mark_instrument("EURUSD", 1.2, now + 2)
    ledger.set_fx_mark("EUR", 1.2, now + 2)
    closed = ledger.snapshot(now + 2)
    assert closed.cash_by_currency["EUR"] == pytest.approx(0)
    assert closed.cash_by_currency["USD"] == pytest.approx(10_098)
    assert closed.positions["EURUSD"].quantity == pytest.approx(0)
    assert closed.realized_pnl == pytest.approx(100)
    assert closed.total_pnl == pytest.approx(98)


def test_missing_or_stale_currency_conversion_fails_closed():
    now = time.time()
    ledger = PortfolioLedger(1_000, max_mark_age_seconds=2)
    ledger.fund_currency("EUR", 100, now, fx_rate_to_reporting=1.1)
    with pytest.raises(ValueError, match="stale EUR"):
        ledger.snapshot(now + 3)

    missing = PortfolioLedger(1_000, max_mark_age_seconds=60)
    missing.register_instrument(make_equity(quote="EUR"))
    assert missing.snapshot(now).positions["AAPL"].quantity == 0
    fill(missing, "euro-stock", "AAPL", "buy", 1, 100, fee=0, fee_currency="EUR", timestamp=now)
    missing.mark_instrument("AAPL", 100, now)
    with pytest.raises(ValueError, match="missing EUR to USD"):
        missing.snapshot(now)


def test_rolling_fx_requires_explicit_margin_and_uses_marked_notional_for_margin():
    now = time.time()
    ledger = PortfolioLedger(10_000, max_mark_age_seconds=60)
    with pytest.raises(ValueError, match="initial_margin_rate"):
        ledger.register_instrument(make_fx(mode=FXAccountingMode.ROLLING))

    fx = make_fx(mode=FXAccountingMode.ROLLING, initial_margin_rate=0.02,
                 maintenance_margin_rate=0.01)
    ledger.register_instrument(fx)
    reserved = ledger.reserve_margin("fx-order", "EURUSD", 10_000, 1.1)
    assert reserved == pytest.approx(220)
    assert ledger.snapshot(now).available_cash_by_currency["USD"] == pytest.approx(9_780)
    ledger.release_reservation("fx-order")

    fill(ledger, "fx-roll-open", "EURUSD", "buy", 10_000, 1.1,
         fee=2, timestamp=now)
    ledger.mark_instrument("EURUSD", 1.2, now + 1)
    status = ledger.snapshot(now + 1)
    assert status.cash_by_currency["USD"] == pytest.approx(9_998)
    assert status.positions["EURUSD"].unrealized_pnl == pytest.approx(1_000)
    assert status.positions["EURUSD"].initial_margin_required == pytest.approx(240)
    assert status.equity == pytest.approx(10_998)

    assert ledger.record_cost("roll-1", "EURUSD", 3, "USD", kind="rollover", timestamp=now + 2)
    assert not ledger.record_cost("roll-1", "EURUSD", 3, "USD", kind="rollover", timestamp=now + 2)
    after_cost = ledger.snapshot(now + 2)
    assert after_cost.costs_paid == pytest.approx(3)
    assert after_cost.total_pnl == pytest.approx(995)


def test_futures_multiplier_variation_margin_fees_and_reservations_reconcile():
    now = time.time()
    ledger = PortfolioLedger(1_000, max_mark_age_seconds=60)
    ledger.register_instrument(make_future())

    assert ledger.reserve_margin("mes-order-a", "MESZ26", 1, 100) == pytest.approx(500)
    with pytest.raises(ValueError, match="reservation exceeds"):
        ledger.reserve_margin("mes-order-b", "MESZ26", 2, 100)
    ledger.release_reservation("mes-order-a")

    fill(ledger, "mes-open", "MESZ26", Side.BUY, 1, 100, fee=1, timestamp=now)
    ledger.mark_instrument("MESZ26", 110, now + 1)
    before_settlement = ledger.snapshot(now + 1)
    assert before_settlement.positions["MESZ26"].unrealized_pnl == pytest.approx(500)
    assert before_settlement.positions["MESZ26"].initial_margin_required == pytest.approx(500)
    assert before_settlement.equity == pytest.approx(1_499)

    assert ledger.settle_future("MESZ26", 110, now + 2) == pytest.approx(500)
    after_settlement = ledger.snapshot(now + 2)
    assert after_settlement.cash_by_currency["USD"] == pytest.approx(1_499)
    assert after_settlement.positions["MESZ26"].realized_pnl == pytest.approx(500)
    assert after_settlement.positions["MESZ26"].unrealized_pnl == pytest.approx(0)
    assert after_settlement.equity == pytest.approx(before_settlement.equity)

    fill(ledger, "mes-close", "MESZ26", Side.SELL, 1, 120, fee=1, timestamp=now + 3)
    closed = ledger.snapshot(now + 3)
    assert closed.positions["MESZ26"].quantity == 0
    assert closed.realized_pnl == pytest.approx(1_000)
    assert closed.fees_paid == pytest.approx(2)
    assert closed.equity == pytest.approx(1_998)
    assert closed.total_pnl == pytest.approx(998)


def test_short_positions_require_borrow_terms_and_accrue_idempotent_costs():
    now = time.time()
    ledger = PortfolioLedger(1_000, max_mark_age_seconds=60)
    with pytest.raises(ValueError, match="borrow_limit"):
        ledger.register_instrument(make_equity(), allow_short=True)

    ledger.register_instrument(make_equity(borrow_limit=5, borrow_fee_rate=0.10), allow_short=True)
    fill(ledger, "short-1", "AAPL", Side.SELL, 2, 100, fee=0, timestamp=now)
    ledger.mark_instrument("AAPL", 90, now + 1)
    with pytest.raises(ValueError, match="borrow_limit"):
        fill(ledger, "short-too-large", "AAPL", Side.SELL, 4, 90,
             fee=0, timestamp=now + 1)

    cost = ledger.accrue_borrow_cost("borrow-day", "AAPL", 0.5, now + 2)
    assert cost == pytest.approx(9)
    assert ledger.accrue_borrow_cost("borrow-day", "AAPL", 0.5, now + 2) == pytest.approx(0)
    ledger.mark_instrument("AAPL", 90, now + 2)
    status = ledger.snapshot(now + 2)
    assert status.positions["AAPL"].quantity == pytest.approx(-2)
    assert status.unrealized_pnl == pytest.approx(20)
    assert status.costs_paid == pytest.approx(9)
    assert status.equity == pytest.approx(1_011)
    assert status.total_pnl == pytest.approx(11)


def test_futures_short_is_margin_governed_and_does_not_require_borrow_terms():
    now = time.time()
    ledger = PortfolioLedger(1_000, max_mark_age_seconds=60)
    ledger.register_instrument(make_future())
    fill(ledger, "mes-short", "MESZ26", Side.SELL, 1, 100, fee=0, timestamp=now)
    ledger.mark_instrument("MESZ26", 90, now + 1)
    status = ledger.snapshot(now + 1)
    assert status.positions["MESZ26"].quantity == pytest.approx(-1)
    assert status.positions["MESZ26"].unrealized_pnl == pytest.approx(500)
    assert status.positions["MESZ26"].initial_margin_required == pytest.approx(500)
    assert status.equity == pytest.approx(1_500)


def test_future_margin_and_accounting_remain_defined_at_zero_and_negative_prices():
    now = time.time()
    ledger = PortfolioLedger(30_000, max_mark_age_seconds=60)
    ledger.register_instrument(make_future())
    assert ledger.reserve_margin("zero-price", "MESZ26", 1, 0) == pytest.approx(500)
    ledger.release_reservation("zero-price")
    fill(ledger, "negative-open", "MESZ26", Side.BUY, 1, -10, fee=1,
         timestamp=now, fill_instrument_id="MESZ26")
    ledger.mark_instrument("MESZ26", -5, now + 1)
    snapshot = ledger.snapshot(now + 1)
    assert snapshot.positions["MESZ26"].unrealized_pnl == pytest.approx(250)
    assert snapshot.positions["MESZ26"].initial_margin_required == pytest.approx(500)


def test_snapshot_fails_closed_on_missing_or_stale_instrument_marks():
    now = time.time()
    ledger = PortfolioLedger(1_000, max_mark_age_seconds=2)
    ledger.register_instrument(make_equity())
    fill(ledger, "open", "AAPL", "buy", 1, 100, fee=0, timestamp=now)
    with pytest.raises(ValueError, match="stale mark"):
        ledger.snapshot(now + 3)
    ledger.mark_instrument("AAPL", 101, now + 1)
    with pytest.raises(ValueError, match="not move backward"):
        ledger.mark_instrument("AAPL", 100, now)


def test_checkpoint_rejects_changed_accounting_metadata_and_preserves_midpoint_marks():
    now = time.time()
    spec = make_equity()
    ledger = PortfolioLedger(10_000, max_mark_age_seconds=60)
    ledger.register_instrument(spec)
    fill(ledger, "aapl-open", "AAPL", Side.BUY, 1, 100.00, fee=0, timestamp=now)
    ledger.mark_instrument("AAPL", 100.005, now + 1)
    checkpoint = ledger.export_state()

    compatible = PortfolioLedger(10_000, max_mark_age_seconds=60)
    compatible.register_instrument(spec)
    compatible.restore_state(checkpoint)
    assert compatible.get_position("AAPL", now + 1).mark_price == pytest.approx(100.005)

    changed = PortfolioLedger(10_000, max_mark_age_seconds=60)
    changed.register_instrument(make_equity(contract_multiplier=2.0))
    with pytest.raises(ValueError, match="accounting metadata changed"):
        changed.restore_state(checkpoint)


def test_checkpoint_allows_covering_an_existing_short_after_borrow_limit_is_tightened():
    now = time.time()
    prior_spec = make_equity(borrow_limit=10, borrow_fee_rate=0.1)
    prior = PortfolioLedger(10_000, max_mark_age_seconds=60)
    prior.register_instrument(prior_spec, allow_short=True)
    fill(prior, "old-short", "AAPL", Side.SELL, 5, 100, timestamp=now)
    checkpoint = prior.export_state()

    tightened = PortfolioLedger(10_000, max_mark_age_seconds=60)
    tightened.register_instrument(
        make_equity(borrow_limit=2, borrow_fee_rate=0.1), allow_short=False)
    tightened.restore_state(checkpoint)
    assert tightened.get_position("AAPL", now).quantity == pytest.approx(-5)
    assert tightened._positions["AAPL"].allow_short is False
    with pytest.raises(ValueError, match="allow_short"):
        fill(tightened, "increase-old-short", "AAPL", Side.SELL, 1, 100, timestamp=now + 1,
             fee=0)
    assert tightened.record_fill(
        "cover-over-limit", "AAPL", Side.BUY, 1, 100, fee=0, fee_currency="USD",
        timestamp=now + 1, allow_risk_reduction=True)
    assert tightened.get_position("AAPL", now + 1).quantity == pytest.approx(-4)


def test_equity_split_and_late_dividend_entitlement_preserve_and_reconcile_equity():
    now = time.time()
    spec = make_equity()
    ledger = PortfolioLedger(10_000, max_mark_age_seconds=60)
    ledger.register_instrument(spec)
    fill(ledger, "split-open", "AAPL", Side.BUY, 10, 100, timestamp=now)
    ledger.mark_instrument("AAPL", 100, now + 1)
    before = ledger.snapshot(now + 1)

    assert ledger.apply_equity_split("aapl-split-2-for-1", "AAPL", 2, now + 2) is True
    split = ledger.snapshot(now + 2)
    assert split.positions["AAPL"].quantity == pytest.approx(20)
    assert split.positions["AAPL"].average_entry_price == pytest.approx(50)
    assert split.positions["AAPL"].mark_price == pytest.approx(50)
    assert split.equity == pytest.approx(before.equity)
    assert ledger.apply_equity_split("aapl-split-2-for-1", "AAPL", 2, now + 10) is False

    fill(ledger, "post-record-sale", "AAPL", Side.SELL, 5, 50, timestamp=now + 3)
    after_sale = ledger.snapshot(now + 3)
    assert after_sale.positions["AAPL"].quantity == pytest.approx(15)
    assert ledger.apply_equity_dividend(
        "aapl-dividend", "AAPL", 0.5, eligible_quantity=20, timestamp=now + 4) == pytest.approx(10)
    dividend = ledger.snapshot(now + 4)
    assert dividend.equity == pytest.approx(split.equity + 10)
    assert dividend.income == pytest.approx(10)
    assert dividend.income_by_currency["USD"] == pytest.approx(10)
    assert dividend.translation_pnl == pytest.approx(0)
    assert ledger.apply_equity_dividend(
        "aapl-dividend", "AAPL", 0.5, eligible_quantity=20, timestamp=now + 10) is None
    with pytest.raises(ValueError, match="reused with different data"):
        ledger.apply_equity_dividend(
            "aapl-dividend", "AAPL", 0.6, eligible_quantity=20, timestamp=now + 10)

    checkpoint = ledger.export_state()
    restored = PortfolioLedger(10_000, max_mark_age_seconds=60)
    restored.register_instrument(spec)
    restored.restore_state(checkpoint)
    assert restored.snapshot(now + 3).equity == pytest.approx(dividend.equity)
    assert restored.apply_equity_dividend(
        "aapl-dividend", "AAPL", 0.5, eligible_quantity=20, timestamp=now + 10) is None


def test_short_equity_dividend_is_debited_and_reported_as_cost():
    now = time.time()
    ledger = PortfolioLedger(10_000, max_mark_age_seconds=60)
    ledger.register_instrument(make_equity(borrow_limit=5, borrow_fee_rate=0), allow_short=True)
    fill(ledger, "short-dividend-open", "AAPL", Side.SELL, 2, 100, timestamp=now)
    ledger.mark_instrument("AAPL", 100, now + 1)
    before = ledger.snapshot(now + 1)
    assert ledger.apply_equity_dividend(
        "short-dividend", "AAPL", 1, eligible_quantity=-2, timestamp=now + 2) == pytest.approx(-2)
    after = ledger.snapshot(now + 2)
    assert after.equity == pytest.approx(before.equity - 2)
    assert after.costs_paid == pytest.approx(2)


def test_rolling_fx_financing_is_signed_idempotent_and_reconciled():
    now = time.time()
    ledger = PortfolioLedger(10_000, max_mark_age_seconds=60)
    ledger.register_instrument(make_fx(
        mode=FXAccountingMode.ROLLING, initial_margin_rate=0.05,
        maintenance_margin_rate=0.025))
    fill(ledger, "fx-financing-open", "EURUSD", Side.BUY, 1000, 1.1, timestamp=now)
    ledger.mark_instrument("EURUSD", 1.1, now + 1)
    before = ledger.snapshot(now + 1)
    assert ledger.record_financing("fx-roll-debit", "EURUSD", -2, "USD",
                                   timestamp=now + 2) is True
    debited = ledger.snapshot(now + 2)
    assert debited.equity == pytest.approx(before.equity - 2)
    assert debited.costs_paid == pytest.approx(2)
    assert ledger.record_financing("fx-roll-debit", "EURUSD", -2, "USD",
                                   timestamp=now + 10) is False
    with pytest.raises(ValueError, match="reused with different data"):
        ledger.record_financing("fx-roll-debit", "EURUSD", -3, "USD",
                                timestamp=now + 10)
    assert ledger.record_financing("fx-roll-credit", "EURUSD", 1, "USD",
                                   timestamp=now + 3) is True
    credited = ledger.snapshot(now + 3)
    assert credited.equity == pytest.approx(debited.equity + 1)
    assert credited.income == pytest.approx(1)
    assert credited.translation_pnl == pytest.approx(0)
    checkpoint = ledger.export_state()
    restored = PortfolioLedger(10_000, max_mark_age_seconds=60)
    restored.register_instrument(make_fx(
        mode=FXAccountingMode.ROLLING, initial_margin_rate=0.05,
        maintenance_margin_rate=0.025))
    restored.restore_state(checkpoint)
    assert restored.record_financing("fx-roll-debit", "EURUSD", -2, "USD",
                                     timestamp=now + 10) is False
    with pytest.raises(ValueError, match="settlement currency"):
        ledger.record_financing("wrong-currency", "EURUSD", -1, "EUR", timestamp=now + 4)


def test_rolling_fx_financing_can_be_posted_after_the_position_is_closed():
    now = time.time()
    ledger = PortfolioLedger(10_000, max_mark_age_seconds=60)
    ledger.register_instrument(make_fx(
        mode=FXAccountingMode.ROLLING, initial_margin_rate=0.05,
        maintenance_margin_rate=0.025))
    fill(ledger, "late-carry-open", "EURUSD", Side.BUY, 1000, 1.1, timestamp=now)
    ledger.mark_instrument("EURUSD", 1.1, now + 1)
    fill(ledger, "late-carry-close", "EURUSD", Side.SELL, 1000, 1.1, timestamp=now + 2)
    assert ledger.snapshot(now + 2).positions["EURUSD"].quantity == pytest.approx(0)
    before = ledger.snapshot(now + 2)

    assert ledger.record_financing(
        "late-settled-carry", "EURUSD", -2, "USD", timestamp=now + 3) is True
    after = ledger.snapshot(now + 3)
    assert after.equity == pytest.approx(before.equity - 2)
    assert after.costs_paid == pytest.approx(2)
    assert ledger.record_financing(
        "late-settled-carry", "EURUSD", -2, "USD", timestamp=now + 10) is False


def test_deliverable_fx_borrow_cost_uses_contract_multiplier():
    now = time.time()
    ledger = PortfolioLedger(10_000, max_mark_age_seconds=60)
    spec = make_fx(contract_multiplier=2, borrow_limit=2_000, borrow_fee_rate=0.1)
    ledger.register_instrument(spec, allow_short=True)
    ledger.set_fx_mark("EUR", 1.1, now, max_age_seconds=60)
    fill(ledger, "multiplier-fx-short", "EURUSD", Side.SELL, 1000, 1.1,
         timestamp=now)
    ledger.mark_instrument("EURUSD", 1.1, now + 1)
    cost = ledger.accrue_borrow_cost("fx-borrow-multiplier", "EURUSD", 0.5, now + 2)
    assert cost == pytest.approx(100)
    status = ledger.snapshot(now + 2)
    assert status.costs_paid == pytest.approx(110)
