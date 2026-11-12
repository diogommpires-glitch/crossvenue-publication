"""Public currency quote wiring and preserved native-currency accounting."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.config.cross_asset_config import CrossAssetConfigurationError, parse_cross_asset_config
from src.connectors.market_data import MarketDataProvider
from src.connectors.provider_factory import (
    build_market_data_provider_factories,
    build_market_data_providers,
)
from src.market.events import MarketEventKind
from src.market.instruments import AssetClass, FXAccountingMode
from src.runtime.application_service import PaperApplicationService
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.recording import SessionRecorder, iter_records, load_manifest
from src.runtime.replay import FaultSpec, ReplayRunner, VirtualClock
from src.runtime.state_manager import StateManager

EXAMPLE = Path(__file__).parents[1] / "config/cross_asset_paper.example.json"
NOW = datetime(2026, 11, 11, 12, tzinfo=UTC).timestamp()
OLD = NOW - 2 * 86400
PAIR = "COINBASE:USDT_USD"
PROVIDER = "coinbase_valuation"


def public_document(*, live_quote=True):
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8-sig"))
    raw.update(starting_capital=10000, starting_balances={"USDT": 10000})
    raw["conversion_marks"] = {
        "USDT": {
            "rate_to_reporting": 1.0,
            "source": "historical_manual_usdt_usd_parity_assumption",
            "max_age_seconds": 86400,
        }
    }
    if live_quote:
        raw["providers"].append(
            {"provider_id": PROVIDER, "type": "coinbase", "instrument_ids": [PAIR]}
        )
        raw["instruments"].append(
            {
                "provider_id": PROVIDER,
                "provider_symbol": "USDT-USD",
                "fill_model": "fx_quote_cross",
                "queue_fraction": 0,
                "maker_latency_ms": 0,
                "enabled": False,
                "instrument": {
                    "instrument_id": PAIR,
                    "asset_class": "fx",
                    "symbol": "USDT-USD",
                    "venue": "COINBASE",
                    "base_currency": "USDT",
                    "quote_currency": "USD",
                    "settlement_currency": "USD",
                    "quantity_unit": "USDT",
                    "tick_size": 0.000001,
                    "quantity_step": 1,
                    "min_quantity": 1,
                    "contract_multiplier": 1,
                    "session_timezone": "UTC",
                    "fx_accounting_mode": "deliverable",
                },
                "strategy": {"order_size": 1, "spread_bps": 8, "inventory_skew_bps": 8},
                "risk": {
                    "max_order_quantity": 1,
                    "max_position_quantity": 1,
                    "max_notional": 10,
                    "allow_short": False,
                },
                "fees": {
                    "maker_rate": 0,
                    "taker_rate": 0,
                    "maker_per_unit": 0,
                    "taker_per_unit": 0,
                    "currency": "USD",
                },
            }
        )
    return raw


def preserved_account(tmp_path):
    old_config = parse_cross_asset_config(public_document(live_quote=False))
    manager = StateManager(str(tmp_path / "valuation.db"))
    old_engine = CrossAssetPaperEngine(old_config, clock=VirtualClock(OLD), state_manager=manager)
    old_engine._save_checkpoint(OLD)
    assert old_engine.ledger.cash_by_currency == {"USDT": 10000}
    config = parse_cross_asset_config(public_document())
    clock = VirtualClock(NOW)
    engine = CrossAssetPaperEngine(
        config, clock=clock, state_manager=manager, allow_config_migration=True
    )
    return config, clock, manager, engine


def public_provider(config, clock):
    from src.connectors.coinbase_provider import CoinbasePublicQuoteProvider

    return CoinbasePublicQuoteProvider(config.instruments[PAIR].spec, receive_clock=clock)


def quote(provider, clock, *, sequence=42, bid=1.001999, ask=1.002001, source_lag=0.5):
    source_time = (
        datetime.fromtimestamp(clock.value - source_lag, UTC).isoformat().replace("+00:00", "Z")
    )
    return provider.normalize_message(
        {
            "bids": [[str(bid), "100", "1"]],
            "asks": [[str(ask), "200", "1"]],
            "sequence": sequence,
            "auction_mode": False,
            "time": source_time,
        },
        receive_ts=clock.value,
    )


def test_coinbase_quote_configuration_is_deliverable_and_disabled():
    old = parse_cross_asset_config(public_document(live_quote=False))
    config = parse_cross_asset_config(public_document())
    item = config.instruments[PAIR]
    assert len(old.instruments) == 4 and len(config.instruments) == 5
    assert item.spec.asset_class == AssetClass.FX
    assert item.spec.fx_accounting_mode == FXAccountingMode.DELIVERABLE
    assert not item.enabled and item.provider_symbol == "USDT-USD"
    assert config.starting_balances == {"USDT": 10000}
    assert config.conversion_marks["USDT"][0] == 1.0


@pytest.mark.parametrize("invalid", ["enabled", "rolling", "venue", "quantity_unit", "multiplier"])
def test_coinbase_configuration_rejects_unsupported_execution_or_contract(invalid):
    raw = public_document()
    item = raw["instruments"][-1]
    if invalid == "enabled":
        item["enabled"] = True
    elif invalid == "rolling":
        item["instrument"].update(
            fx_accounting_mode="rolling", initial_margin_rate=0.05, maintenance_margin_rate=0.025
        )
    else:
        field, value = {
            "venue": ("venue", "OTHER"),
            "quantity_unit": ("quantity_unit", "USDC"),
            "multiplier": ("contract_multiplier", 2),
        }[invalid]
        item["instrument"][field] = value
    with pytest.raises(
        CrossAssetConfigurationError, match="(?i)coinbase.*(disabled|deliverable|valuation)"
    ):
        parse_cross_asset_config(raw)


def test_coinbase_factory_constructs_only_the_matching_read_only_public_provider():
    config = parse_cross_asset_config(public_document())

    class NoCredentials(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("public Coinbase factory must not read credentials")

    factories = build_market_data_provider_factories(config, environ=NoCredentials())
    binding_id, factory, ids = next(row for row in factories if row[0] == PROVIDER)
    provider = factory()
    assert binding_id == PROVIDER and ids == (PAIR,)
    assert isinstance(provider, MarketDataProvider)
    assert provider.instrument == config.instruments[PAIR].spec
    assert provider.capabilities.quote and not provider.capabilities.trades
    assert provider.capabilities.order_book
    assert not any(
        hasattr(provider, name)
        for name in ("place_order", "cancel_order", "get_balance", "authenticate")
    )
    selected = replace(config, providers={PROVIDER: config.providers[PROVIDER]})
    eager = build_market_data_providers(selected, environ=NoCredentials())
    assert [(identity, instruments) for identity, _provider, instruments in eager] == [
        (PROVIDER, (PAIR,))
    ]


def test_live_bid_ask_values_native_usdt_without_converting_or_resetting_cash(tmp_path):
    config, clock, _manager, engine = preserved_account(tmp_path)
    with pytest.raises(ValueError, match="stale USDT"):
        engine.ledger.snapshot(NOW)
    provider = public_provider(config, clock)
    event = quote(provider, clock)
    assert event.kind == MarketEventKind.BOOK_SNAPSHOT and event.source == "coinbase"
    assert event.event_ts == NOW - 0.5 and event.receive_ts == NOW
    assert engine.on_market_event(event, provider_id=PROVIDER, generation=1)
    portfolio = engine.ledger.snapshot(NOW)
    assert portfolio.cash_by_currency == {"USDT": 10000}
    assert portfolio.positions[PAIR].quantity == 0
    assert portfolio.equity == pytest.approx(10020)
    assert portfolio.total_pnl == portfolio.translation_pnl == pytest.approx(20)
    assert portfolio.realized_pnl == portfolio.unrealized_pnl == portfolio.fees_paid == 0
    engine.start_strategy()
    assert engine.is_running
    assert engine._instrument_readiness_code(engine._runtime[PAIR], NOW) == "quoting_disabled"
    assert engine.total_fills == 0 and engine.get_active_orders_detail() == []
    with pytest.raises(ValueError, match="fresh open market data"):
        engine.submit_paper_order(PAIR, "BUY", 1, 1.0, request_id="disabled")
    assert engine.ledger.cash_by_currency == {"USDT": 10000}


def test_quote_expiry_duplicate_and_restart_do_not_revive_historical_parity(tmp_path):
    config, clock, manager, engine = preserved_account(tmp_path)
    provider = public_provider(config, clock)
    event = quote(provider, clock)
    assert engine.on_market_event(event, provider_id=PROVIDER, generation=1)
    clock.advance_to(NOW + 5)
    assert not engine.on_market_event(event, provider_id=PROVIDER, generation=1)
    assert engine.ledger.export_state()["fx_marks"]["USDT"][1] == NOW
    engine.invalidate_provider(PROVIDER, "synthetic socket loss", generation=2)
    clock.advance_to(NOW + 31)
    with pytest.raises(ValueError, match="stale USDT"):
        engine.ledger.snapshot(clock.value)
    assert engine.ledger.cash_by_currency == {"USDT": 10000}
    engine._save_checkpoint(clock.value)
    restored = CrossAssetPaperEngine(config, clock=clock, state_manager=manager)
    with pytest.raises(ValueError, match="stale USDT"):
        restored.ledger.snapshot(clock.value)
    saved_mark = restored.ledger.export_state()["fx_marks"]["USDT"]
    assert saved_mark == [pytest.approx(1.002), NOW, 30]
    assert restored.ledger.cash_by_currency == {"USDT": 10000}
    clock.advance_to(NOW + 32)
    new_event = quote(provider, clock, sequence=43, bid=0.998999, ask=0.999001)
    assert not engine.on_market_event(new_event, provider_id=PROVIDER, generation=1)
    assert engine.on_market_event(new_event, provider_id=PROVIDER, generation=2)
    assert restored.on_market_event(new_event, provider_id=PROVIDER, generation=2)
    assert restored.ledger.snapshot(clock.value).equity == pytest.approx(9990)
    assert restored.ledger.cash_by_currency == {"USDT": 10000}


def test_source_admission_and_receive_valuation_age_keep_the_existing_time_convention(tmp_path):
    config, clock, _manager, engine = preserved_account(tmp_path)
    provider = public_provider(config, clock)
    event = quote(provider, clock, source_lag=12)
    assert event.event_ts == NOW - 12
    assert engine.on_market_event(event, provider_id=PROVIDER, generation=1)
    assert engine._runtime[PAIR].last_quote_event_ts == NOW - 12
    assert engine.ledger.export_state()["fx_marks"]["USDT"][1] == NOW
    assert engine.ledger.snapshot(NOW + 30).equity == pytest.approx(10020)
    with pytest.raises(ValueError, match="stale USDT"):
        engine.ledger.snapshot(NOW + 30.001)


def test_recorded_live_valuation_replays_identical_financial_state_without_schema_change(tmp_path):
    config, clock, _manager, engine = preserved_account(tmp_path)
    provider = public_provider(config, clock)
    directory = tmp_path / "recording"
    recorder = SessionRecorder(directory, config=public_document())
    recorder.set_initial_state(engine._checkpoint_payload())
    service = PaperApplicationService(engine, recorder=recorder, clock=clock)
    service.start()
    try:
        for sequence, offset, bid, ask in (
            (42, 0, 1.001999, 1.002001),
            (43, 2, 0.998999, 0.999001),
        ):
            clock.advance_to(NOW + offset)
            assert service.enqueue_market(
                PROVIDER, quote(provider, clock, sequence=sequence, bid=bid, ask=ask), generation=1
            )
            service.flush()
        expected = engine.ledger.snapshot(clock.value)
    finally:
        service.stop()
    manifest = load_manifest(directory)
    assert manifest["schema_version"] == 2
    events = [row for row in iter_records(directory) if row["kind"] == "market"]
    assert len(events) == 2
    assert {row["payload"]["kind"] for row in events} == {"book_snapshot"}
    assert all(row["payload"]["event_ts"] == row["payload"]["receive_ts"] - 0.5 for row in events)
    first = ReplayRunner(config).run(directory)
    duplicated = ReplayRunner(config).run(directory, faults=FaultSpec(duplicate_every=1))
    expected_financial = {
        "cash": dict(expected.cash_by_currency),
        "positions": {
            key: {
                "quantity": position.quantity,
                "average_entry_price": position.average_entry_price,
                "realized_pnl": position.realized_pnl,
            }
            for key, position in expected.positions.items()
        },
        "equity": expected.equity,
        "total_pnl": expected.total_pnl,
        "fees_paid": expected.fees_paid,
        "costs_paid": expected.costs_paid,
        "income": expected.income,
        "valuation_valid": True,
    }
    assert first["financial"] == expected_financial
    assert first["financial"] == duplicated["financial"]
    assert first["digest"] == duplicated["digest"]
    assert first["financial"]["cash"] == expected.cash_by_currency == {"USDT": 10000}
    assert first["financial"]["equity"] == pytest.approx(expected.equity)
    assert first["financial"]["total_pnl"] == pytest.approx(expected.translation_pnl)
    assert first["fills"] == [] and first["input_complete"]
