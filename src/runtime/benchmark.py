"""Measured offline workload through the real account writer.

Default runs are finite. The 24-hour soak is an explicit duration, never a
claim inferred from a short accelerated test. No provider network is used.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import platform
import tempfile
import time

from ..config.cross_asset_config import ProviderConfig, load_cross_asset_config
from ..market.events import MarketDataEvent, MarketEventKind
from ..utils.types import Side
from .replay import _ReplayProvider


def benchmark_config(count: int = 25, config_path=None):
    if count <= 0:
        raise ValueError("instrument count must be positive")
    path = Path(config_path) if config_path else Path(__file__).parents[2] / "config" / "cross_asset_paper.example.json"
    source = load_cross_asset_config(path)
    prototypes = list(source.instruments.values())
    instruments = {}
    for index in range(count):
        original = prototypes[index % len(prototypes)]
        identity = "BENCH:{}:{:03d}".format(original.spec.asset_class.value, index)
        spec = replace(original.spec, instrument_id=identity, symbol="B{:03d}".format(index))
        instruments[identity] = replace(original, spec=spec, provider_id="benchmark",
                                        provider_symbol=spec.symbol)
    return replace(source, account_id="offline_benchmark", starting_capital=10_000_000,
        starting_balances={"USD": 9_000_000.0, "USDT": 1_000_000.0},
        max_daily_loss=1_000_000, max_gross_exposure=1_000_000_000,
        max_orders_per_minute=1_000_000, instruments=instruments,
        providers={"benchmark": ProviderConfig("benchmark", "binance", tuple(instruments))})


def run_benchmark(*, instrument_count: int = 25, event_rate: int = 1000,
                  sustained_seconds: float = 10, burst_rate: int = 5000,
                  burst_seconds: float = 30, soak_seconds: float = 0,
                  config_path=None):
    if min(event_rate, burst_rate) <= 0 or min(sustained_seconds, burst_seconds, soak_seconds) < 0:
        raise ValueError("rates must be positive and durations non-negative")
    from .application_service import PaperApplicationService
    from .cross_asset_paper_engine import CrossAssetPaperEngine
    from .portfolio_analytics import PortfolioAnalytics
    from .state_manager import StateManager
    config = benchmark_config(instrument_count, config_path)
    temporary = tempfile.TemporaryDirectory(prefix="paper-benchmark-")
    database = Path(temporary.name) / "benchmark.db"
    manager = StateManager(str(database))
    writer = manager.acquire_account_writer(config.account_id)
    acquired = False
    service = None
    try:
        writer.__enter__()
        acquired = True
        engine = CrossAssetPaperEngine(config, state_manager=manager)
        engine.register_provider("benchmark", _ReplayProvider("benchmark"), config.instruments)
        analytics = PortfolioAnalytics(manager, config.account_id)
        service = PaperApplicationService(engine, state_manager=manager, queue_capacity=10000, analytics=analytics)
        phases = []
        sequence = {key: 0 for key in config.instruments}
        prices = {"crypto": 80000.0, "equity": 200.0, "fx": 1.1, "future": 5000.0}
        items = list(config.instruments.items())
        service.start()
        now = time.time()
        for identity, item in items:
            service.enqueue_market("benchmark", MarketDataEvent(identity, "offline-benchmark",
                MarketEventKind.STATUS, now, now, status="tradeable"))
            mid, tick = prices[item.spec.asset_class.value], item.spec.tick_size
            service.enqueue_market("benchmark", MarketDataEvent(identity, "offline-benchmark", MarketEventKind.BOOK_SNAPSHOT,
                now, now, bids=((item.spec.round_price(mid - tick, Side.BUY), 100000.0),),
                asks=((item.spec.round_price(mid + tick, Side.SELL), 100000.0),)))
            if item.fill_model == "trade_print_queue":
                service.enqueue_market("benchmark", MarketDataEvent(identity, "offline-benchmark", MarketEventKind.TRADE,
                    now, now, price=item.spec.round_price(mid), quantity=100.0, side=Side.BUY))
        service.call(engine.start_strategy)
        # Explicit warmup performs initial subscriptions/quote commits outside
        # steady-state measurements; the report says this rather than hiding it.
        service.call(lambda: service._delays.clear())
        for label, duration, rate in (("sustained", sustained_seconds, event_rate),
                                      ("burst", burst_seconds, burst_rate),
                                      ("soak", soak_seconds, event_rate)):
            if duration == 0:
                continue
            if hasattr(service, 'reset_latency_metrics'):
                service.call(service.reset_latency_metrics)
            started = time.perf_counter()
            sent = accepted = 0
            due = started
            last_tick = started
            while time.perf_counter() - started < duration:
                identity, item = items[sent % len(items)]
                sequence[identity] += 1
                mid = prices[item.spec.asset_class.value]
                tick = item.spec.tick_size
                stamp = time.time()
                if sequence[identity] % 3 == 0 and item.fill_model == "trade_print_queue":
                    price = item.spec.round_price(mid * (0.999 if sequence[identity] % 2 else 1.001))
                    event = MarketDataEvent(identity, "offline-benchmark", MarketEventKind.TRADE,
                        stamp, stamp, price=price, quantity=100.0,
                        side=Side.SELL if sequence[identity] % 2 else Side.BUY)
                else:
                    event = MarketDataEvent(identity, "offline-benchmark", MarketEventKind.BOOK_SNAPSHOT,
                        stamp, stamp, bids=((item.spec.round_price(mid - tick, Side.BUY), 100000.0),),
                        asks=((item.spec.round_price(mid + tick, Side.SELL), 100000.0),))
                accepted += bool(service.enqueue_market("benchmark", event))
                sent += 1
                due += 1.0 / rate
                current = time.perf_counter()
                if current - last_tick >= 1:
                    service.tick()
                    last_tick = current
                if due > current:
                    time.sleep(due - current)
            production_elapsed = time.perf_counter() - started
            service.flush(timeout=max(30.0, duration))
            elapsed = time.perf_counter() - started
            state = service.get_dashboard_state()
            phases.append({"phase": label, "requested_duration_seconds": duration,
                           "actual_duration_seconds": elapsed, "target_events_per_second": rate,
                           "generated_events": sent, "accepted_events": accepted,
                           "producer_elapsed_seconds": production_elapsed,
                           "generated_events_per_second": sent / production_elapsed,
                           "application": state.get("application", {})})
        command = {"protocol_version": 1, "request_id": "benchmark-global-stop",
                   "account_id": config.account_id, "expected_revision": manager.get_account_revision(config.account_id),
                   "command": "stop_strategy", "payload": {}}
        before_stop = time.perf_counter()
        response = service.submit_command(command).result(timeout=10)
        stop_seconds = time.perf_counter() - before_stop
        service.flush(timeout=30)
        final = service.get_dashboard_state()
        storage = sum(path.stat().st_size for path in database.parent.glob("benchmark.db*"))
        return {"schema_version": 1, "kind": "offline_measured_benchmark",
                "reference_machine": {"platform": platform.platform(), "python": platform.python_version(),
                                      "logical_processors": os.cpu_count()},
                "instrument_count": instrument_count, "asset_classes": sorted({item.spec.asset_class.value for _, item in items}),
                "phases": phases, "stop_acknowledgment_seconds": stop_seconds,
                "stop_response": response, "database_bytes": storage,
                "final_application": final.get("application", {}),
                "paper_fills": engine.total_fills,
                "warmup_excluded_from_queue_latency": True,
                "soak_target_seconds": soak_seconds,
                "soak_actual_wall_seconds": sum(phase["actual_duration_seconds"] for phase in phases if phase["phase"] == "soak"),
                "dashboard_browser_freshness": "unmeasured_requires_browser_acceptance",
                "live_acceptance": "pending"}
    finally:
        if service is not None:
            service.stop()
        if acquired:
            writer.__exit__(None, None, None)
        temporary.cleanup()
