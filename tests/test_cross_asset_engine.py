import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.config.cross_asset_config import load_cross_asset_config
from src.connectors.market_data import ProviderCapabilities
from src.market.events import MarketDataEvent, MarketEventKind
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.state_manager import StateManager
from src.runtime.websocket_ui_server import WebSocketUIServer
from src.orderbook.order import Order
from src.utils.types import OrderType, Price, Quantity, Side, Timestamp


CONFIG_PATH = Path(__file__).parents[1] / "config" / "cross_asset_paper.example.json"


class Clock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def make_engine(clock, state_manager=None, config=None, max_clock_skew_seconds=0):
    config = config or load_cross_asset_config(CONFIG_PATH)
    return CrossAssetPaperEngine(
        config,
        clock=clock,
        max_clock_skew_seconds=max_clock_skew_seconds,
        state_manager=state_manager,
    )


def btc_book(timestamp, sequence=1):
    return MarketDataEvent(
        instrument_id="BINANCE:BTCUSDT", source="fixture",
        kind=MarketEventKind.BOOK_SNAPSHOT,
        event_ts=timestamp, receive_ts=timestamp, sequence=sequence,
        bids=((79999.99, 1.0),), asks=((80000.01, 1.0),),
    )


def btc_trade(timestamp, sequence, price, side=Side.BUY):
    return MarketDataEvent(
        instrument_id="BINANCE:BTCUSDT", source="fixture",
        kind=MarketEventKind.TRADE,
        event_ts=timestamp, receive_ts=timestamp, sequence=sequence,
        price=price, quantity=0.01, side=side,
    )


def test_trade_prints_before_order_acceptance_cannot_fill_and_fills_are_local():
    now = time.time()
    clock = Clock(now)
    engine = make_engine(clock)
    engine.start_strategy()
    assert engine.on_market_event(btc_book(now))
    # This public print establishes trade readiness; it predates the quotes.
    assert engine.on_market_event(btc_trade(now - 1, 1, 80000.0))
    accepted = engine.get_active_orders_detail()
    buy = next(order for order in accepted if order["side"] == "BUY")
    assert buy["instrument_id"] == "BINANCE:BTCUSDT"

    # A delayed pre-order event is delivered after acceptance but retains its
    # original source event time, so it cannot fill the new quote.
    clock.value = now + 0.5
    delayed = MarketDataEvent(
        instrument_id="BINANCE:BTCUSDT", source="fixture", kind=MarketEventKind.TRADE,
        event_ts=now - 0.1, receive_ts=clock.value, sequence=2,
        price=float(buy["price"]), quantity=0.01, side=Side.SELL,
    )
    assert engine.on_market_event(delayed)
    assert engine.total_fills == 0

    clock.value = now + 1.0
    assert engine.on_market_event(btc_trade(clock.value, 3, float(buy["price"]) - 0.01, Side.SELL))
    assert engine.total_fills == 1
    fill = engine.trades[-1]
    assert fill["execution_model"] == "trade_print_queue_estimate"
    assert fill["estimated"] is True
    assert fill["price"] == pytest.approx(float(buy["price"]))
    assert fill["source"] == "fixture"
    assert fill["instrument_id"] == "BINANCE:BTCUSDT"
    assert fill["realized_pnl"] == pytest.approx(0)
    assert fill["realized_pnl_currency"] == "USDT"
    assert fill["total_pnl"] is not None


def test_state_manager_recovery_preserves_cross_asset_paper_ledger_and_history(tmp_path):
    now = time.time()
    clock = Clock(now)
    manager = StateManager(str(tmp_path / "paper.db"))
    engine = make_engine(clock, manager)
    engine.start_strategy()
    engine.on_market_event(btc_book(now))
    engine.on_market_event(btc_trade(now - 1, 1, 80000.0))
    buy = next(order for order in engine.get_active_orders_detail() if order["side"] == "BUY")
    clock.value = now + 1.0
    engine.on_market_event(btc_trade(clock.value, 2, float(buy["price"]), Side.SELL))
    expected = engine.ledger.snapshot(as_of=clock.value)
    engine.stop_strategy()

    restarted = make_engine(clock, manager)
    recovered = restarted.ledger.snapshot(as_of=clock.value)
    assert recovered.equity == pytest.approx(expected.equity)
    assert recovered.positions["BINANCE:BTCUSDT"].quantity == pytest.approx(0.001)
    assert restarted.total_fills == 1
    assert restarted.trades[-1]["execution_model"] == "trade_print_queue_estimate"
    assert restarted.get_active_orders_detail() == []
    assert restarted.get_dashboard_state()["execution_assumptions_verified"] is True


def test_checkpoint_rejects_changed_fee_or_fill_assumptions(tmp_path):
    now = time.time()
    clock = Clock(now)
    manager = StateManager(str(tmp_path / "paper-assumptions.db"))
    engine = make_engine(clock, manager)
    engine._save_checkpoint(now)
    config = load_cross_asset_config(CONFIG_PATH)
    instruments = dict(config.instruments)
    btc = instruments["BINANCE:BTCUSDT"]
    instruments["BINANCE:BTCUSDT"] = replace(
        btc, fees=replace(btc.fees, maker_rate=btc.fees.maker_rate * 2))
    changed_config = replace(config, instruments=instruments)
    with pytest.raises(ValueError, match="fee or fill-model assumptions changed"):
        make_engine(clock, manager, changed_config)


def test_strategy_parameter_update_is_scoped_to_the_selected_instrument():
    engine = make_engine(Clock(time.time()))
    engine.update_parameters({"instrument_id": "NASDAQ:AAPL", "spread_bps": 17})
    assert engine._runtime["NASDAQ:AAPL"].strategy.spread_bps == pytest.approx(17)
    assert engine._runtime["BINANCE:BTCUSDT"].strategy.spread_bps == pytest.approx(8)
    assert engine._runtime["OANDA:EUR_USD"].strategy.spread_bps == pytest.approx(8)


def test_runtime_quote_settings_survive_restart_without_marking_history_unverified(tmp_path):
    now = time.time()
    clock = Clock(now)
    manager = StateManager(str(tmp_path / "paper-strategy-settings.db"))
    engine = make_engine(clock, manager)
    engine.update_parameters({
        "instrument_id": "NASDAQ:AAPL", "spread_bps": 17,
        "inventory_skew_bps": 11,
    })

    restarted = make_engine(clock, manager)
    strategy = restarted._runtime["NASDAQ:AAPL"].strategy
    assert strategy.spread_bps == pytest.approx(17)
    assert strategy.inventory_skew_bps == pytest.approx(11)
    assert restarted.get_dashboard_state()["execution_assumptions_verified"] is True


def test_source_timestamp_lead_beyond_configured_bound_is_rejected():
    now = time.time()
    engine = make_engine(Clock(now), max_clock_skew_seconds=3)
    rejected = []
    engine.on("market_data_rejected", rejected.append)

    assert engine.on_market_event(MarketDataEvent(
        instrument_id="BINANCE:BTCUSDT", source="bad-clock-fixture",
        kind=MarketEventKind.BOOK_SNAPSHOT,
        event_ts=now + 4.0, receive_ts=now,
        bids=((79999.99, 1.0),), asks=((80000.01, 1.0),),
    )) is False
    assert rejected[-1]["reason"] == "source_clock_skew_exceeds_configured_bound"


def test_repeated_fx_status_messages_do_not_churn_quotes_before_eligible_print():
    now = time.time()
    clock = Clock(now)
    engine = make_engine(clock, max_clock_skew_seconds=3)
    engine.start_strategy()
    instrument_id = "OANDA:EUR_USD"
    source = "oanda-status-fixture"

    def book(receive, event, ask):
        return MarketDataEvent(
            instrument_id=instrument_id, source=source,
            kind=MarketEventKind.BOOK_SNAPSHOT,
            event_ts=event, receive_ts=receive,
            bids=((round(ask - 0.00010, 5), 20000),), asks=((ask, 20000),),
        )

    def status(receive, event):
        return MarketDataEvent(
            instrument_id=instrument_id, source=source,
            kind=MarketEventKind.STATUS, event_ts=event, receive_ts=receive,
            status="tradeable",
        )

    assert engine.on_market_event(book(now, now - 1, 1.10010))
    assert engine.on_market_event(status(now, now - 1))
    buy = next(order for order in engine.get_active_orders_detail()
               if order["instrument_id"] == instrument_id and order["side"] == "BUY")

    for offset in (1.0, 2.0, 3.0, 4.0):
        clock.value = now + offset
        assert engine.on_market_event(book(clock.value, clock.value - 1, 1.10010))
        assert engine.on_market_event(status(clock.value, clock.value - 1))
        current = next(order for order in engine.get_active_orders_detail()
                       if order["instrument_id"] == instrument_id and order["side"] == "BUY")
        assert current["order_id"] == buy["order_id"]

    clock.value = now + 4.51
    crossing_ask = round(float(buy["price"]) - engine.config.instruments[instrument_id].spec.tick_size, 5)
    assert engine.on_market_event(book(clock.value, now + 3.51, crossing_ask))
    assert engine.total_fills == 1
    assert engine.trades[-1]["instrument_id"] == instrument_id


def test_backdated_fx_financing_checks_risk_at_current_engine_time():
    now = time.time()
    clock = Clock(now)
    engine = make_engine(clock)
    engine.ledger.record_fill(
        "risk-time-fx-open", "OANDA:EUR_USD", Side.BUY, 1000, 1.1,
        fee=0, fee_currency="USD", timestamp=now)
    engine.ledger.record_fill(
        "risk-time-aapl-open", "NASDAQ:AAPL", Side.BUY, 1, 200,
        fee=0, fee_currency="USD", timestamp=now)
    engine.ledger.mark_instrument("OANDA:EUR_USD", 1.1, now)
    engine.ledger.mark_instrument("NASDAQ:AAPL", 200, now)
    clock.value = now + 20
    engine.ledger.mark_instrument("OANDA:EUR_USD", 1.1, clock.value)
    engine.ledger.mark_instrument("NASDAQ:AAPL", 200, clock.value)

    assert engine.record_fx_financing(
        "backdated-risk-time-carry", "OANDA:EUR_USD", -1, "USD",
        timestamp=now + 2) is True
    assert engine._last_account_timestamp == pytest.approx(clock.value)


def test_backdated_fx_financing_advances_recovery_snapshot_and_survives_restart(tmp_path):
    now = time.time()
    clock = Clock(now)
    manager = StateManager(str(tmp_path / "late-financing-recovery.db"))
    engine = make_engine(clock, manager)

    # Model a newer, already persisted checkpoint followed by financing that
    # is processed late but retains its original accounting timestamp.
    engine._save_checkpoint(now + 20)
    clock.value = now + 20
    assert engine.record_fx_financing(
        "restart-late-carry", "OANDA:EUR_USD", -2, "USD",
        timestamp=now + 2) is True

    checkpoint = manager.load_cross_asset_snapshot(engine.account_id)
    assert checkpoint["timestamp"] == pytest.approx(now + 20)
    assert checkpoint["snapshot"]["ledger"]["cash"]["USD"] == pytest.approx(39998)

    restarted = make_engine(clock, manager)
    recovered = restarted.ledger.snapshot(as_of=clock.value)
    assert recovered.cash_by_currency["USD"] == pytest.approx(39998)
    assert recovered.costs_paid_by_currency["USD"] == pytest.approx(2)


def test_provider_clock_ahead_delayed_fx_quote_cannot_fill_before_local_acceptance():
    now = time.time()
    clock = Clock(now)
    engine = make_engine(clock, max_clock_skew_seconds=3)
    instrument_id = "OANDA:EUR_USD"
    source = "ahead-clock-fixture"
    assert engine.on_market_event(MarketDataEvent(
        instrument_id=instrument_id, source=source, kind=MarketEventKind.BOOK_SNAPSHOT,
        event_ts=now + 2.0, receive_ts=now,
        bids=((1.10000, 20000),), asks=((1.10010, 20000),),
    ))
    assert engine.on_market_event(MarketDataEvent(
        instrument_id=instrument_id, source=source, kind=MarketEventKind.STATUS,
        event_ts=now + 2.0, receive_ts=now, status="tradeable",
    ))
    engine.start_strategy()
    buy = next(order for order in engine.get_active_orders_detail()
               if order["instrument_id"] == instrument_id and order["side"] == "BUY")

    # This print timestamp is ahead of local time but still within the
    # configured source-clock skew allowance; its receive time is after order
    # acceptance, while its source time is not beyond the conservative bound.
    clock.value = now + 0.5
    delayed_ask = round(float(buy["price"]) - 0.00001, 5)
    assert engine.on_market_event(MarketDataEvent(
        instrument_id=instrument_id, source=source, kind=MarketEventKind.BOOK_SNAPSHOT,
        event_ts=now + 2.5, receive_ts=clock.value,
        bids=((round(delayed_ask - 0.00010, 5), 20000),), asks=((delayed_ask, 20000),),
    ))
    assert engine.total_fills == 0

    clock.value = now + 1.0
    later_ask = round(float(buy["price"]) - 0.00001, 5)
    assert engine.on_market_event(MarketDataEvent(
        instrument_id=instrument_id, source=source, kind=MarketEventKind.BOOK_SNAPSHOT,
        event_ts=now + 4.0, receive_ts=clock.value,
        bids=((round(later_ask - 0.00010, 5), 20000),), asks=((later_ask, 20000),),
    ))
    assert engine.total_fills == 1
    assert engine.trades[-1]["execution_model"] == "fx_quote_cross_estimate"


def test_lagged_feed_quotes_survive_until_a_post_cutoff_trade_can_arrive():
    now = time.time()
    clock = Clock(now)
    engine = make_engine(clock, max_clock_skew_seconds=3)
    engine.start_strategy()
    instrument_id = "BINANCE:BTCUSDT"
    source = "lagged-clock-fixture"

    # A one-second source-to-receive lag has been observed on this feed.
    assert engine.on_market_event(MarketDataEvent(
        instrument_id=instrument_id, source=source, kind=MarketEventKind.BOOK_SNAPSHOT,
        event_ts=now - 1, receive_ts=now,
        bids=((79999.99, 1.0),), asks=((80000.01, 1.0),),
    ))
    assert engine.on_market_event(MarketDataEvent(
        instrument_id=instrument_id, source=source, kind=MarketEventKind.TRADE,
        event_ts=now - 1, receive_ts=now, sequence=1,
        price=80000.0, quantity=0.01, side=Side.BUY,
    ))
    buy = next(order for order in engine.get_active_orders_detail()
               if order["instrument_id"] == instrument_id and order["side"] == "BUY")

    # Regular refresh opportunities must not churn the quote before the
    # accepted order's event-time cutoff plus the observed delivery delay.
    for offset in (1.0, 2.0, 3.0, 4.0):
        clock.value = now + offset
        assert engine.on_market_event(MarketDataEvent(
            instrument_id=instrument_id, source=source,
            kind=MarketEventKind.BOOK_SNAPSHOT,
            event_ts=clock.value - 1, receive_ts=clock.value,
            bids=((79999.99, 1.0),), asks=((80000.01, 1.0),),
        ))
        current = next(order for order in engine.get_active_orders_detail()
                       if order["instrument_id"] == instrument_id and order["side"] == "BUY")
        assert current["order_id"] == buy["order_id"]

    # The print's source timestamp is beyond acceptance + configured skew;
    # it arrives after the source-to-receive delay and fills the retained
    # local paper quote at its resting limit price.
    clock.value = now + 4.5
    assert engine.on_market_event(MarketDataEvent(
        instrument_id=instrument_id, source=source, kind=MarketEventKind.TRADE,
        event_ts=now + 3.5, receive_ts=clock.value, sequence=2,
        price=float(buy["price"]), quantity=0.01, side=Side.SELL,
    ))
    assert engine.total_fills == 1
    assert engine.trades[-1]["price"] == pytest.approx(float(buy["price"]))


def test_oanda_fx_quote_fill_is_explicitly_estimated_and_not_a_trade_print():
    now = time.time()
    clock = Clock(now)
    engine = make_engine(clock)
    engine.start_strategy()
    instrument_id = "OANDA:EUR_USD"
    event = MarketDataEvent(
        instrument_id=instrument_id, source="oanda-fixture",
        kind=MarketEventKind.BOOK_SNAPSHOT, event_ts=now, receive_ts=now,
        bids=((1.10000, 20000),), asks=((1.10010, 20000),),
    )
    assert engine.on_market_event(event)
    status = MarketDataEvent(
        instrument_id=instrument_id, source="oanda-fixture",
        kind=MarketEventKind.STATUS, event_ts=now, receive_ts=now, status="tradeable",
    )
    assert engine.on_market_event(status)
    order = next(order for order in engine.get_active_orders_detail() if order["instrument_id"] == instrument_id
                 and order["side"] == "BUY")

    clock.value = now + 1.0
    spec = engine.config.instruments[instrument_id].spec
    tick = spec.tick_size
    ask = spec.round_price(float(order["price"]) - tick)
    bid = spec.round_price(ask - tick)
    moved_quote = MarketDataEvent(
        instrument_id=instrument_id, source="oanda-fixture",
        kind=MarketEventKind.BOOK_SNAPSHOT, event_ts=clock.value, receive_ts=clock.value,
        bids=((bid, 20000),), asks=((ask, 20000),),
    )
    assert engine.on_market_event(moved_quote)
    fx_fill = next(fill for fill in engine.trades if fill["instrument_id"] == instrument_id)
    assert fx_fill["estimated"] is True
    assert fx_fill["execution_model"] == "fx_quote_cross_estimate"
    assert fx_fill["source"] == "oanda-fixture"
    assert all(trade["instrument_id"] != instrument_id for trade in engine._runtime[instrument_id].recent_public_trades)


def test_dashboard_full_state_exposes_instruments_and_currency_portfolio():
    engine = make_engine(Clock(time.time()))
    server = WebSocketUIServer(bridge=engine)

    class Socket:
        sent = None

        async def send(self, raw):
            self.sent = raw

    socket = Socket()
    # This mock websocket send completes immediately; stepping the coroutine
    # avoids creating a platform event loop for a purely synchronous payload test.
    send_state = server.send_full_state(socket)
    with pytest.raises(StopIteration):
        send_state.send(None)
    payload = json.loads(socket.sent)
    assert payload["multi_asset_state"]["schema_version"] == 2
    assert {item["asset_class"] for item in payload["multi_asset_state"]["instruments"]} == {
        "crypto", "equity", "fx", "future"
    }
    assert payload["multi_asset_state"]["portfolio"]["equity"] == pytest.approx(50000)


def test_provider_secret_fields_and_string_boolean_are_rejected(tmp_path):
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["risk"]["api_key"] = "should-not-be-embedded"
    path = tmp_path / "secrets.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="secret field"):
        load_cross_asset_config(path)

    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["instruments"][1]["risk"]["allow_short"] = "false"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON boolean"):
        load_cross_asset_config(path)


def test_concurrent_risk_checks_cannot_both_spend_the_same_gross_limit():
    now = time.time()
    clock = Clock(now)
    config = replace(load_cross_asset_config(CONFIG_PATH), max_gross_exposure=250.0)
    engine = make_engine(clock, config=config)
    orders = [
        Order(order_id="btc-risk", side=Side.BUY, order_type=OrderType.LIMIT,
              quantity=Quantity(0.001), price=Price(80000.0), timestamp=Timestamp(now),
              trader_id="test", instrument_id="BINANCE:BTCUSDT"),
        Order(order_id="aapl-risk", side=Side.BUY, order_type=OrderType.LIMIT,
              quantity=Quantity(1), price=Price(200.0), timestamp=Timestamp(now),
              trader_id="test", instrument_id="NASDAQ:AAPL"),
    ]
    original_projection = engine._projected_gross_exposure

    def yield_during_risk_check(*args, **kwargs):
        value = original_projection(*args, **kwargs)
        time.sleep(0.05)
        return value

    engine._projected_gross_exposure = yield_during_risk_check
    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = list(pool.map(
            lambda order: engine._accept_order(engine._runtime[order.instrument_id], order, now),
            orders,
        ))

    assert sum(accepted) == 1
    snapshot = engine.ledger.snapshot(as_of=now)
    assert engine._projected_gross_exposure(snapshot, now) <= 250.0 + 1e-9


def test_tightened_position_limit_still_allows_a_strictly_reducing_order():
    now = time.time()
    clock = Clock(now)
    config = load_cross_asset_config(CONFIG_PATH)
    instruments = dict(config.instruments)
    btc = instruments["BINANCE:BTCUSDT"]
    instruments["BINANCE:BTCUSDT"] = replace(btc, max_position_quantity=0.001)
    config = replace(config, instruments=instruments)
    engine = make_engine(clock, config=config)
    engine.ledger.record_fill("legacy-over-limit", "BINANCE:BTCUSDT", Side.BUY,
                              0.02, 80_000.0, fee=0, fee_currency="USDT", timestamp=now)
    close = Order(
        order_id="close-over-limit", side=Side.SELL, order_type=OrderType.LIMIT,
        quantity=Quantity(0.001), price=Price(80_000.01), timestamp=Timestamp(now),
        trader_id="risk-reduction", instrument_id="BINANCE:BTCUSDT")

    assert engine._accept_order(engine._runtime["BINANCE:BTCUSDT"], close, now) is True
    assert engine.get_active_orders_detail()[0]["side"] == "SELL"


def test_strictly_reducing_close_bypasses_saturated_order_rate_limit():
    now = time.time()
    clock = Clock(now)
    config = replace(load_cross_asset_config(CONFIG_PATH), max_orders_per_minute=1)
    engine = make_engine(clock, config=config)
    engine.ledger.record_fill(
        "rate-limit-seed-long", "BINANCE:BTCUSDT", Side.BUY, 0.001, 80_000,
        fee=0, fee_currency="USDT", timestamp=now)
    engine.ledger.mark_instrument("BINANCE:BTCUSDT", 80_000, now)
    engine._orders_per_minute.append(now)

    close = Order(
        order_id="rate-limited-close", side=Side.SELL, order_type=OrderType.LIMIT,
        quantity=Quantity(0.001), price=Price(80_000.01), timestamp=Timestamp(now),
        trader_id="risk-reduction", instrument_id="BINANCE:BTCUSDT")
    increase = Order(
        order_id="rate-limited-increase", side=Side.BUY, order_type=OrderType.LIMIT,
        quantity=Quantity(0.001), price=Price(80_000.0), timestamp=Timestamp(now),
        trader_id="risk-increase", instrument_id="BINANCE:BTCUSDT")
    rejected = []
    engine.on("order_rejected", rejected.append)

    assert engine._accept_order(engine._runtime["BINANCE:BTCUSDT"], close, now) is True
    assert engine._accept_order(engine._runtime["BINANCE:BTCUSDT"], increase, now) is False
    assert rejected[-1]["reason"] == "portfolio_order_rate_limit"


def test_strict_close_remains_available_when_an_unrelated_position_mark_is_stale():
    now = time.time()
    clock = Clock(now)
    engine = make_engine(clock)
    engine.ledger.record_fill(
        "stale-unrelated-btc-long", "BINANCE:BTCUSDT", Side.BUY, 0.001, 80_000,
        fee=0, fee_currency="USDT", timestamp=now)
    engine.ledger.mark_instrument("BINANCE:BTCUSDT", 80_000, now)
    engine.ledger.record_fill(
        "stale-unrelated-aapl-long", "NASDAQ:AAPL", Side.BUY, 1, 200,
        fee=0, fee_currency="USD", timestamp=now - 100)
    engine.ledger.mark_instrument("NASDAQ:AAPL", 200, now - 100)

    close = Order(
        order_id="close-with-stale-portfolio", side=Side.SELL, order_type=OrderType.LIMIT,
        quantity=Quantity(0.001), price=Price(80_000.01), timestamp=Timestamp(now),
        trader_id="risk-reduction", instrument_id="BINANCE:BTCUSDT")
    assert engine._accept_order(engine._runtime["BINANCE:BTCUSDT"], close, now) is True


def test_unrealized_loss_limits_new_margin_risk_before_maintenance_breach():
    now = time.time()
    clock = Clock(now)
    config = load_cross_asset_config(CONFIG_PATH)
    instruments = dict(config.instruments)
    instruments["CME:ESZ6"] = replace(
        instruments["CME:ESZ6"], max_position_quantity=2)
    config = replace(
        config, starting_capital=40_000.0, starting_balances={"USD": 40_000.0},
        max_daily_loss=100_000.0, instruments=instruments)
    engine = make_engine(clock, config=config)
    engine.ledger.record_fill("margin-boundary-open", "CME:ESZ6", Side.BUY, 1,
                              5000.0, fee=0, fee_currency="USD", timestamp=now)
    engine.ledger.mark_instrument("CME:ESZ6", 4980.0, now + 1)
    clock.value = now + 1
    snapshot = engine.ledger.snapshot(as_of=clock.value)
    assert snapshot.equity == pytest.approx(39_000)
    assert snapshot.maintenance_margin_required == pytest.approx(18_000)
    assert snapshot.available_cash_by_currency["USD"] == pytest.approx(20_000)

    order = Order(
        order_id="second-es-contract", side=Side.BUY, order_type=OrderType.LIMIT,
        quantity=Quantity(1), price=Price(4980.0), timestamp=Timestamp(clock.value),
        trader_id="risk-test", instrument_id="CME:ESZ6")
    rejected = []
    engine.on("order_rejected", rejected.append)
    assert engine._accept_order(engine._runtime["CME:ESZ6"], order, clock.value) is False
    assert rejected[-1]["reason"] == "projected_initial_margin_exceeds_equity"


def test_gross_cap_allows_closing_quote_and_futures_margin_call_is_non_fabricating():
    # This margin scenario must remain before the example contract's real expiry.
    now = datetime(2026, 10, 20, 12, tzinfo=timezone.utc).timestamp()
    clock = Clock(now)
    config = load_cross_asset_config(CONFIG_PATH)
    instruments = dict(config.instruments)
    es = instruments["CME:ESZ6"]
    instruments["CME:ESZ6"] = replace(es, max_position_quantity=2)
    config = replace(
        config,
        starting_capital=20_000.0,
        starting_balances={"USD": 20_000.0},
        max_daily_loss=100_000.0,
        instruments=instruments,
    )
    engine = make_engine(clock, config=config)
    engine.ledger.record_fill("es-open", "CME:ESZ6", Side.BUY, 1, 5000.0,
                              fee=0, fee_currency="USD", timestamp=now)

    def es_book(timestamp, sequence, price):
        return MarketDataEvent(
            instrument_id="CME:ESZ6", source="fixture", kind=MarketEventKind.BOOK_SNAPSHOT,
            event_ts=timestamp, receive_ts=timestamp, sequence=sequence,
            bids=((price - 0.25, 10),), asks=((price + 0.25, 10),),
        )

    assert engine.on_market_event(es_book(now, 1, 5000.0))
    assert engine.on_market_event(MarketDataEvent(
        instrument_id="CME:ESZ6", source="fixture", kind=MarketEventKind.STATUS,
        event_ts=now, receive_ts=now, status="trading",
    ))
    assert engine.on_market_event(MarketDataEvent(
        instrument_id="CME:ESZ6", source="fixture", kind=MarketEventKind.TRADE,
        event_ts=now, receive_ts=now, sequence=1, price=5000.0,
        quantity=2, side=Side.BUY,
    ))

    # The account is using its entire initial margin. The risk-increasing bid
    # cannot reserve another contract, but the quote that closes the long can.
    engine.start_strategy()
    active = engine.get_active_orders_detail()
    assert [order["side"] for order in active if order["instrument_id"] == "CME:ESZ6"] == ["SELL"]
    engine.stop_strategy()

    clock.value = now + 2.0
    assert engine.on_market_event(es_book(clock.value, 2, 4500.0))
    engine.expire_stale_feeds(clock.value)
    assert engine.circuit_breaker_active is True
    assert engine.circuit_breaker_reason == "maintenance_margin_breach"
    assert engine.ledger.get_position("CME:ESZ6", clock.value).quantity == pytest.approx(1)

    # Restarting the strategy while halted may expose a reducing paper quote,
    # but it does not invent a forced liquidation fill.
    engine.start_strategy()
    active = engine.get_active_orders_detail()
    assert any(order["instrument_id"] == "CME:ESZ6" and order["side"] == "SELL"
               for order in active)
    assert engine.ledger.get_position("CME:ESZ6", clock.value).quantity == pytest.approx(1)


def test_risk_halt_clamps_close_quote_to_a_partial_position():
    now = time.time()
    clock = Clock(now)
    engine = make_engine(clock)
    instrument_id = "BINANCE:BTCUSDT"
    engine.ledger.record_fill("partial-position", instrument_id, Side.BUY, 0.0005,
                              80_000.0, fee=0, fee_currency="USDT", timestamp=now)
    assert engine.on_market_event(btc_book(now))
    assert engine.on_market_event(btc_trade(now, 1, 80_000.0))
    engine._trip_circuit_breaker("test_halt", now)

    engine.start_strategy()
    active = [order for order in engine.get_active_orders_detail()
              if order["instrument_id"] == instrument_id]
    assert [(order["side"], order["quantity"]) for order in active] == [("SELL", 0.0005)]
    row = next(item for item in engine.get_dashboard_state()["instruments"]
               if item["instrument_id"] == instrument_id)
    assert row["risk_reduction_status"] == "reduce_only"


def test_daily_loss_gap_uses_last_known_equity_instead_of_resetting_baseline():
    before_midnight = datetime(2026, 10, 24, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    after_midnight = datetime(2026, 10, 25, 0, 1, 0, tzinfo=timezone.utc).timestamp()
    clock = Clock(before_midnight)
    engine = make_engine(clock)
    previous_close = engine.ledger.snapshot(as_of=before_midnight)
    assert engine._daily_loss_breached(previous_close, before_midnight) is False

    engine.ledger.record_cost(
        "overnight-gap", "BINANCE:BTCUSDT", 600, "USD", kind="test-gap",
        timestamp=after_midnight,
    )
    clock.value = after_midnight
    current = engine.ledger.snapshot(as_of=after_midnight)
    assert engine._daily_loss_breached(current, after_midnight) is True


def test_partial_provider_start_failure_stops_the_provider_that_raised():
    engine = make_engine(Clock(time.time()))

    class PartiallyStartedProvider:
        capabilities = ProviderCapabilities(quote=True, order_book=True, trades=True)

        def __init__(self):
            self.stopped = False

        def start(self, on_event):
            raise RuntimeError("transport allocated, then start failed")

        def stop(self):
            self.stopped = True

    provider = PartiallyStartedProvider()
    engine.register_provider("partial", provider, ("BINANCE:BTCUSDT",))
    with pytest.raises(RuntimeError, match="start failed"):
        engine.start_providers()
    assert provider.stopped is True
