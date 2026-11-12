"""Small, explicitly synthetic four-market recordings for offline acceptance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..config.cross_asset_config import load_cross_asset_config
from ..market.events import MarketDataEvent, MarketEventKind
from ..utils.types import Side
from .recording import SessionRecorder


def create_representative_fixture(directory, config_path=None, *, seconds: int = 120,
                                  include_faults: bool = True):
    if seconds < 10:
        raise ValueError("fixture needs at least ten seconds")
    config_path = Path(config_path) if config_path else Path(__file__).parents[2] / "config" / "cross_asset_paper.example.json"
    config = load_cross_asset_config(config_path)
    import json
    raw_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    base = 1792497600.0
    prices = {"crypto": 80000.0, "equity": 200.0, "fx": 1.1, "future": 5000.0}
    generations = {key: 0 for key in config.providers}
    with SessionRecorder(directory, config=raw_config, code_version="synthetic-fixture-v1", seed=7,
                         source_kind="synthetic_fixture") as recorder:
        recorder.append("command", {"command": "start_strategy", "payload": {}}, base)
        for offset in range(seconds + 1):
            stamp = base + offset
            for instrument_id, item in config.instruments.items():
                spec = item.spec
                mid = prices[spec.asset_class.value]
                if spec.asset_class.value == "fx":
                    mid *= 1 + (-0.002 if offset % 12 >= 6 else 0.002)
                tick = spec.tick_size
                bid, ask = spec.round_price(mid - tick, Side.BUY), spec.round_price(mid + tick, Side.SELL)
                source = "fixture:" + item.provider_id
                status = "halted" if include_faults and offset == 30 else "tradeable"
                if offset == 0 or (include_faults and offset in (30, 31)):
                    recorder.record_market(MarketDataEvent(instrument_id, source, MarketEventKind.STATUS,
                        stamp, stamp, status=status), provider_id=item.provider_id,
                        generation=generations[item.provider_id])
                if item.fill_model == "trade_print_queue":
                    price = spec.round_price(mid * (1.01 if offset % 2 else 0.99))
                    recorder.record_market(MarketDataEvent(instrument_id, source, MarketEventKind.TRADE,
                        stamp, stamp, sequence=offset + 1, price=price, quantity=100.0,
                        side=Side.BUY if offset % 2 else Side.SELL), provider_id=item.provider_id,
                        generation=generations[item.provider_id])
                book = MarketDataEvent(instrument_id, source, MarketEventKind.BOOK_SNAPSHOT,
                    stamp, stamp, sequence=offset + 1, bids=((bid, 100000.0),), asks=((ask, 100000.0),))
                recorder.record_market(book, provider_id=item.provider_id, generation=generations[item.provider_id])
                if include_faults and offset == 20:
                    recorder.record_market(book, provider_id=item.provider_id, generation=generations[item.provider_id])
                if include_faults and offset == 40:
                    recorder.record_market(replace(book, event_ts=stamp - 2), provider_id=item.provider_id,
                                           generation=generations[item.provider_id])
            if include_faults and offset == 50:
                provider_id = next(iter(config.providers))
                recorder.append("provider", {"state": "invalidated", "reason": "synthetic_disconnect"},
                                stamp, provider_id=provider_id)
                generations[provider_id] = 1
                recorder.append("provider", {"state": "healthy"}, stamp, provider_id=provider_id, generation=1)
            recorder.append("timer", {"action": "maintenance"}, stamp)
        recorder.append("command", {"command": "stop_strategy", "payload": {}}, base + seconds)
    return Path(directory)
