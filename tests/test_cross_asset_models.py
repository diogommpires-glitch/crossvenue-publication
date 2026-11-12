from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, ROUND_UP

import pytest

from src.market.events import MarketDataEvent, MarketEventKind
from src.market.instruments import AssetClass, FXAccountingMode, InstrumentSpec
from src.utils.types import Side


def equity_spec(**overrides):
    values = {
        "instrument_id": "NASDAQ:AAPL",
        "asset_class": AssetClass.EQUITY,
        "symbol": "AAPL",
        "venue": "NASDAQ",
        "base_currency": "usd",
        "quote_currency": "usd",
        "settlement_currency": "usd",
        "quantity_unit": "shares",
        "tick_size": 0.01,
        "quantity_step": 1.0,
        "min_quantity": 1.0,
        "capabilities": {"market_data", "paper_execution"},
    }
    values.update(overrides)
    return InstrumentSpec(**values)


def test_instrument_spec_normalizes_currencies_and_freezes_capabilities():
    spec = equity_spec()

    assert spec.base_currency == "USD"
    assert spec.quote_currency == "USD"
    assert spec.settlement_currency == "USD"
    assert spec.capabilities == frozenset({"market_data", "paper_execution"})
    with pytest.raises(AttributeError):
        spec.symbol = "MSFT"


def test_side_aware_tick_rounding_and_quantity_steps():
    spec = equity_spec(tick_size=0.05, quantity_step=0.1, min_quantity=0.1)

    assert spec.round_price(100.03, Side.BUY) == pytest.approx(100.0)
    assert spec.round_price(100.03, Side.SELL) == pytest.approx(100.05)
    assert spec.round_price(100.025) == pytest.approx(100.05)
    assert spec.round_quantity(1.29) == pytest.approx(1.2)
    assert spec.round_quantity(1.21, rounding=ROUND_UP) == pytest.approx(1.3)
    assert spec.round_quantity(1.25, rounding=ROUND_HALF_UP) == pytest.approx(1.3)
    assert spec.is_valid_price(100.05)
    assert not spec.is_valid_price(100.03)
    assert spec.is_valid_quantity(1.2)
    assert not spec.is_valid_quantity(1.25)


def test_fx_and_future_have_product_specific_required_metadata():
    fx = InstrumentSpec(
        instrument_id="OANDA:EUR_USD",
        asset_class=AssetClass.FX,
        symbol="EUR_USD",
        venue="OANDA",
        base_currency="EUR",
        quote_currency="USD",
        settlement_currency="USD",
        quantity_unit="base_units",
        tick_size=0.00001,
        quantity_step=1,
        min_quantity=1,
        fx_accounting_mode=FXAccountingMode.ROLLING,
    )
    future = InstrumentSpec(
        instrument_id="XCME:ESZ26",
        asset_class=AssetClass.FUTURE,
        symbol="ESZ26",
        venue="XCME",
        base_currency="ES",
        quote_currency="USD",
        settlement_currency="USD",
        quantity_unit="contracts",
        tick_size=0.25,
        quantity_step=1,
        min_quantity=1,
        contract_multiplier=50,
        expiry=datetime(2026, 12, 18, 21, 0, tzinfo=timezone.utc),
        initial_margin_per_contract=20_000,
        maintenance_margin_per_contract=18_000,
    )

    assert fx.fx_accounting_mode == FXAccountingMode.ROLLING
    assert future.expiry.tzinfo == timezone.utc
    assert future.round_price(-0.13, Side.BUY) == pytest.approx(-0.25)


@pytest.mark.parametrize("overrides, message", [
    ({"tick_size": 0}, "tick_size must be positive"),
    ({"quantity_step": float("nan")}, "quantity_step must be a finite number"),
    ({"min_quantity": 0.5}, "min_quantity must be aligned"),
    ({"contract_multiplier": -1}, "contract_multiplier must be positive"),
    ({"fx_accounting_mode": FXAccountingMode.DELIVERABLE}, "only valid for FX"),
    ({"capabilities": "short"}, "iterable of capability names"),
])
def test_instrument_spec_rejects_invalid_metadata(overrides, message):
    with pytest.raises(ValueError, match=message):
        equity_spec(**overrides)


def test_fx_mode_future_expiry_and_margin_invariants_are_enforced():
    with pytest.raises(ValueError, match="require an FXAccountingMode"):
        InstrumentSpec(
            instrument_id="FX:EUR_USD", asset_class=AssetClass.FX, symbol="EUR_USD",
            venue="OANDA", base_currency="EUR", quote_currency="USD",
            settlement_currency="USD", quantity_unit="base_units", tick_size=0.00001,
            quantity_step=1, min_quantity=1,
        )

    with pytest.raises(ValueError, match="require an expiry"):
        InstrumentSpec(
            instrument_id="XCME:ESH27", asset_class=AssetClass.FUTURE, symbol="ESH27",
            venue="XCME", base_currency="ES", quote_currency="USD",
            settlement_currency="USD", quantity_unit="contracts", tick_size=0.25,
            quantity_step=1, min_quantity=1, initial_margin_per_contract=20_000,
            maintenance_margin_per_contract=18_000,
        )

    with pytest.raises(ValueError, match="timezone-aware"):
        equity_spec(expiry=datetime(2026, 1, 1))

    with pytest.raises(ValueError, match="cannot exceed"):
        InstrumentSpec(
            instrument_id="XCME:ESH27", asset_class=AssetClass.FUTURE, symbol="ESH27",
            venue="XCME", base_currency="ES", quote_currency="USD",
            settlement_currency="USD", quantity_unit="contracts", tick_size=0.25,
            quantity_step=1, min_quantity=1,
            expiry=datetime(2027, 3, 19, tzinfo=timezone.utc),
            initial_margin_per_contract=1_000, maintenance_margin_per_contract=1_500,
        )

    with pytest.raises(ValueError, match="settlement_currency must match quote_currency"):
        InstrumentSpec(
            instrument_id="FX:EUR_USD", asset_class=AssetClass.FX, symbol="EUR_USD",
            venue="OANDA", base_currency="EUR", quote_currency="USD",
            settlement_currency="EUR", quantity_unit="base_units", tick_size=0.00001,
            quantity_step=1, min_quantity=1, fx_accounting_mode=FXAccountingMode.ROLLING,
            initial_margin_rate=0.05, maintenance_margin_rate=0.025,
        )


def test_rounding_rejects_nonfinite_bad_side_and_below_minimum():
    spec = equity_spec()

    with pytest.raises(ValueError, match="finite"):
        spec.round_price(float("inf"))
    with pytest.raises(ValueError, match="side must"):
        spec.round_price(1.0, "buy")
    with pytest.raises(ValueError, match="below min_quantity"):
        spec.round_quantity(0.5)
    with pytest.raises(ValueError, match="positive"):
        spec.round_quantity(0)


def test_market_event_normalizes_books_and_keeps_timestamps_distinct():
    event = MarketDataEvent(
        instrument_id="NASDAQ:AAPL",
        source="databento",
        kind=MarketEventKind.BOOK_SNAPSHOT,
        event_ts=10.0,
        receive_ts=10.2,
        sequence=42,
        bids=((99.0, 2.0), (100.0, 1.0)),
        asks=((102.0, 3.0), (101.0, 4.0)),
    )

    assert event.bids == ((100.0, 1.0), (99.0, 2.0))
    assert event.asks == ((101.0, 4.0), (102.0, 3.0))
    assert event.event_ts == 10.0
    assert event.receive_ts == 10.2
    with pytest.raises(AttributeError):
        event.source = "other"


def test_market_event_trade_requires_price_and_positive_quantity():
    trade = MarketDataEvent(
        instrument_id="OANDA:EUR_USD", source="oanda", kind=MarketEventKind.TRADE,
        event_ts=0.0, receive_ts=0.0, price=1.085, quantity=1000.0, side=Side.BUY,
    )
    assert trade.side == Side.BUY

    with pytest.raises(ValueError, match="require price and positive quantity"):
        MarketDataEvent(
            instrument_id="OANDA:EUR_USD", source="oanda", kind=MarketEventKind.TRADE,
            event_ts=0.0, receive_ts=0.0, quantity=1000.0,
        )
    with pytest.raises(ValueError, match="must be positive"):
        MarketDataEvent(
            instrument_id="OANDA:EUR_USD", source="oanda", kind=MarketEventKind.TRADE,
            event_ts=0.0, receive_ts=0.0, price=1.0, quantity=0.0,
        )


def test_market_event_allows_negative_future_price_and_zero_delta_deletion():
    delta = MarketDataEvent(
        instrument_id="XCME:CLZ26", source="databento", kind=MarketEventKind.BOOK_DELTA,
        event_ts=1.0, receive_ts=1.1, sequence=100,
        bids=((-1.0, 0.0),), asks=((-0.5, 2.0),),
    )
    assert delta.bids == ((-1.0, 0.0),)
    assert delta.asks == ((-0.5, 2.0),)


@pytest.mark.parametrize("kwargs, message", [
    ({"instrument_id": " "}, "instrument_id must be a non-empty string"),
    ({"event_ts": float("nan")}, "event_ts must be finite"),
    ({"sequence": True}, "sequence must be a non-negative integer"),
    ({"kind": "trade"}, "kind must be a MarketEventKind"),
    ({"side": "buy"}, "side must be a Side"),
    ({"bids": ((100.0, 0.0),)}, "bids level quantity must be positive"),
])
def test_market_event_rejects_invalid_envelopes(kwargs, message):
    values = {
        "instrument_id": "NASDAQ:AAPL", "source": "databento",
        "kind": MarketEventKind.BOOK_SNAPSHOT, "event_ts": 1.0, "receive_ts": 1.1,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        MarketDataEvent(**values)


def test_session_and_status_events_require_their_normalized_values():
    with pytest.raises(ValueError, match="require a session value"):
        MarketDataEvent(
            instrument_id="XCME:ESZ26", source="databento", kind=MarketEventKind.SESSION,
            event_ts=1.0, receive_ts=1.0,
        )
    with pytest.raises(ValueError, match="require a status value"):
        MarketDataEvent(
            instrument_id="XCME:ESZ26", source="databento", kind=MarketEventKind.STATUS,
            event_ts=1.0, receive_ts=1.0,
        )

