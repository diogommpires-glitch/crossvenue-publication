"""Read-only adapter from Binance public crypto streams to canonical events."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Optional

from ..market.events import MarketDataEvent, MarketEventKind
from ..market.instruments import AssetClass, InstrumentSpec
from ..utils.types import Side
from .binance_websocket import BinanceWebSocket
from .market_data import ProviderCapabilities, ProviderState, ProviderStatusMixin


class BinancePublicDataProvider(ProviderStatusMixin):
    """Normalize public Binance depth and trade streams; expose no execution API."""

    name = "binance"

    def __init__(
        self,
        instrument: InstrumentSpec,
        symbol: Optional[str] = None,
        *,
        connector: Any = None,
        connector_factory: Callable[..., Any] = BinanceWebSocket,
        receive_clock: Callable[[], float] = time.time,
    ) -> None:
        ProviderStatusMixin.__init__(self)
        if not isinstance(instrument, InstrumentSpec) or instrument.asset_class != AssetClass.CRYPTO:
            raise ValueError("BinancePublicDataProvider requires a crypto InstrumentSpec")
        if not callable(receive_clock):
            raise TypeError("receive_clock must be callable")
        self.instrument = instrument
        self.symbol = str(symbol or instrument.symbol).strip().lower()
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        self._connector = connector
        self._connector_factory = connector_factory
        self._receive_clock = receive_clock

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(quote=True, order_book=True, trades=True, sessions=False, status=True)

    def start(self, on_event: Callable[[MarketDataEvent], None]) -> None:
        self._set_event_handler(on_event)
        if self._connector is None:
            self._connector = self._connector_factory(
                symbol=self.symbol,
                on_orderbook=self._on_book,
                on_trade=self._on_trade,
            )
        else:
            callbacks = getattr(self._connector, "callbacks", None)
            if isinstance(callbacks, dict):
                callbacks.update(on_orderbook=self._on_book, on_trade=self._on_trade)
            else:
                if hasattr(self._connector, "on_orderbook"):
                    self._connector.on_orderbook = self._on_book
                if hasattr(self._connector, "on_trade"):
                    self._connector.on_trade = self._on_trade
        self._set_provider_state(ProviderState.CONNECTING, "Starting Binance public market-data streams")
        try:
            self._connector.start()
        except Exception as exc:
            self._set_provider_error("Binance public market-data connection failed: {}".format(exc))
            raise
        stats = self.status_snapshot()
        if stats.get("depth_connected") or stats.get("trades_connected"):
            self._set_provider_state(ProviderState.CONNECTED, "Binance public stream connected")

    def _event_time(self, raw: Optional[float], received: float) -> float:
        if raw is None:
            return received
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return received
        return value if math.isfinite(value) and value >= 0 else received

    def _on_book(self, bids, asks, metadata=None) -> None:
        received = float(self._receive_clock())
        metadata = metadata or {}
        try:
            event = MarketDataEvent(
                instrument_id=self.instrument.instrument_id,
                source=self.name,
                kind=MarketEventKind.BOOK_SNAPSHOT,
                event_ts=self._event_time(metadata.get("event_timestamp"), received),
                receive_ts=received,
                sequence=metadata.get("sequence"),
                bids=tuple((float(price), float(size)) for price, size in bids),
                asks=tuple((float(price), float(size)) for price, size in asks),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self._set_provider_error("Invalid Binance order-book update: {}".format(exc))
            return
        self._publish(event)
        self._set_provider_state(ProviderState.CONNECTED, "Receiving Binance public depth")

    def _on_trade(self, price, quantity, side, event_timestamp=None, trade_id=None) -> None:
        received = float(self._receive_clock())
        try:
            normalized_side = Side.BUY if str(side).lower() == "buy" else Side.SELL
            sequence = None if trade_id is None else int(trade_id)
            event = MarketDataEvent(
                instrument_id=self.instrument.instrument_id,
                source=self.name,
                kind=MarketEventKind.TRADE,
                event_ts=self._event_time(event_timestamp, received),
                receive_ts=received,
                sequence=sequence,
                price=float(price),
                quantity=float(quantity),
                side=normalized_side,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self._set_provider_error("Invalid Binance public trade update: {}".format(exc))
            return
        self._publish(event)
        self._set_provider_state(ProviderState.CONNECTED, "Receiving Binance public trades")

    def status_snapshot(self) -> dict:
        connector = self._connector
        if connector is None:
            return {}
        get_stats = getattr(connector, "get_stats", None)
        return dict(get_stats()) if callable(get_stats) else {}

    def stop(self) -> None:
        if self._connector is not None:
            self._connector.stop()
        self._set_provider_state(ProviderState.STOPPED, "Binance public data stopped")
