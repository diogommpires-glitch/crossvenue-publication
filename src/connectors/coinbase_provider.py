"""Public Coinbase USDT/USD level-1 quotes for deliverable FX valuation."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any

import requests

from ..market.events import MarketDataEvent, MarketEventKind
from ..market.instruments import AssetClass, FXAccountingMode, InstrumentSpec
from .market_data import (
    MarketEventHandler,
    ProviderCapabilities,
    ProviderState,
    ProviderStatusMixin,
)


class CoinbasePublicQuoteProvider(ProviderStatusMixin):
    """Poll one public book; no credentials, account routes or execution API."""

    name = "coinbase"
    URL = "https://api.exchange.coinbase.com/products/USDT-USD/book"
    HTTP_TIMEOUT = (3.0, 3.0)
    BODY_DEADLINE_SECONDS = 6.0
    MAX_RESPONSE_BYTES = 64 * 1024

    def __init__(
        self,
        instrument: InstrumentSpec,
        symbol: str = "USDT-USD",
        *,
        session: Any = None,
        receive_clock: Callable[[], float] = time.time,
        poll_seconds: float = 1.0,
        max_source_age_seconds: float = 10.0,
        max_clock_skew_seconds: float = 3.0,
    ) -> None:
        super().__init__()
        if (
            not isinstance(instrument, InstrumentSpec)
            or instrument.asset_class != AssetClass.FX
            or instrument.fx_accounting_mode != FXAccountingMode.DELIVERABLE
            or instrument.base_currency != "USDT"
            or instrument.quote_currency != "USD"
            or instrument.settlement_currency != "USD"
            or instrument.quantity_unit != "USDT"
            or instrument.venue != "COINBASE"
            or instrument.contract_multiplier != 1
            or symbol != "USDT-USD"
        ):
            raise ValueError(
                "Coinbase quotes require deliverable USDT/USD metadata and symbol USDT-USD"
            )
        if not callable(receive_clock):
            raise TypeError("receive_clock must be callable")
        for name, value in (
            ("poll_seconds", poll_seconds),
            ("max_source_age_seconds", max_source_age_seconds),
            ("max_clock_skew_seconds", max_clock_skew_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
            if name != "max_clock_skew_seconds" and value == 0:
                raise ValueError(f"{name} must be positive")
        self.instrument = instrument
        self.symbol = symbol
        self._session = session
        self.receive_clock = receive_clock
        self.poll_seconds = poll_seconds
        self.max_source_age_seconds = max_source_age_seconds
        self.max_clock_skew_seconds = max_clock_skew_seconds
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._last_sequence: int | None = None
        self._last_source_time: float | None = None

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quote=True, order_book=True, trades=False, sessions=False, status=False
        )

    @staticmethod
    def _timestamp(value: object) -> float:
        if not isinstance(value, str):
            raise ValueError("Coinbase book requires an RFC3339 publisher time")
        match = re.fullmatch(
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})", value
        )
        if match is None:
            raise ValueError("Coinbase book requires an RFC3339 publisher time with timezone")
        seconds, fraction, zone = match.groups()
        if zone != "Z" and (int(zone[1:3]) > 23 or int(zone[4:6]) > 59):
            raise ValueError("Coinbase publisher timezone has invalid hours or minutes")
        stamp = datetime.fromisoformat(seconds + ("+00:00" if zone == "Z" else zone)).timestamp()
        # Preserve nanosecond publisher fractions up to the canonical float's
        # epoch-second precision, rather than substituting local receipt time.
        if fraction:
            stamp += int(fraction) / 10 ** len(fraction)
        if not math.isfinite(stamp) or stamp < 0:
            raise ValueError("Coinbase publisher time must be a finite epoch time")
        return stamp

    @staticmethod
    def _level(value: object, side: str) -> tuple[tuple[float, float], ...]:
        if not isinstance(value, list) or len(value) != 1:
            raise ValueError(f"Coinbase {side} must contain exactly one level-1 quote")
        level = value[0]
        if not isinstance(level, (list, tuple)) or len(level) != 3:
            raise ValueError(f"Coinbase {side} level must contain price, size and order count")
        try:
            if isinstance(level[0], bool) or isinstance(level[1], bool):
                raise ValueError("boolean is not a price or size")
            price, size = float(level[0]), float(level[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Coinbase {side} price and size must be numeric") from exc
        if not math.isfinite(price) or not math.isfinite(size) or price <= 0 or size <= 0:
            raise ValueError(f"Coinbase {side} price and size must be finite and positive")
        return ((price, size),)

    def normalize_message(
        self, payload: Mapping[str, object], receive_ts: float | None = None
    ) -> MarketDataEvent:
        if not isinstance(payload, Mapping):
            raise ValueError("Coinbase book payload must be an object")
        if payload.get("auction_mode") is not False:
            raise ValueError("Coinbase auction mode must be explicitly false")
        sequence = payload.get("sequence")
        if type(sequence) is not int or sequence < 0:
            raise ValueError("Coinbase book sequence must be a non-negative integer")
        received = self.receive_clock() if receive_ts is None else receive_ts
        if (
            isinstance(received, bool)
            or not isinstance(received, (int, float))
            or not math.isfinite(received)
            or received < 0
        ):
            raise ValueError("Coinbase receive time must be a finite epoch time")
        stamp = self._timestamp(payload.get("time"))
        if stamp > received + self.max_clock_skew_seconds:
            raise ValueError("Coinbase publisher time exceeds configured future clock-skew bound")
        if received - stamp > self.max_source_age_seconds + self.max_clock_skew_seconds:
            raise ValueError("Coinbase publisher time exceeds configured source-age bound")
        bids, asks = (
            self._level(payload.get("bids"), "bids"),
            self._level(payload.get("asks"), "asks"),
        )
        if bids[0][0] > asks[0][0]:
            raise ValueError("Coinbase quote is crossed")
        return MarketDataEvent(
            instrument_id=self.instrument.instrument_id,
            source=self.name,
            kind=MarketEventKind.BOOK_SNAPSHOT,
            event_ts=stamp,
            receive_ts=received,
            sequence=sequence,
            bids=bids,
            asks=asks,
        )

    def _request_quote(self, session: Any) -> MarketDataEvent:
        started = time.monotonic()
        response = session.get(
            self.URL,
            params={"level": 1},
            headers={"Accept": "application/json"},
            timeout=self.HTTP_TIMEOUT,
            allow_redirects=False,
            stream=True,
        )
        try:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError(
                    "Coinbase public quote request requires HTTP 200 without redirects"
                )
            body = bytearray()
            # A one-byte read yields control even for a trickling response, so
            # the body deadline is not postponed until a large chunk is filled.
            for chunk in response.iter_content(chunk_size=1):
                if self._stop_event.is_set() and threading.current_thread() is self._worker:
                    raise InterruptedError("Coinbase quote worker stopped")
                if time.monotonic() - started > self.BODY_DEADLINE_SECONDS:
                    raise requests.Timeout("Coinbase quote response exceeded total body deadline")
                body.extend(chunk)
                if len(body) > self.MAX_RESPONSE_BYTES:
                    raise ValueError("Coinbase quote response exceeds 64 KiB")
            received = self.receive_clock()
            try:
                payload = json.loads(body)
            except RecursionError as exc:
                raise ValueError("Coinbase quote JSON exceeds supported nesting depth") from exc
            return self.normalize_message(payload, receive_ts=received)
        finally:
            response.close()

    def fetch_quote(self) -> MarketDataEvent:
        """Fetch and validate one public snapshot without publishing it."""
        if self._session is not None:
            return self._request_quote(self._session)
        with requests.Session() as session:
            session.trust_env = False  # Never load ambient .netrc credentials for a public feed.
            return self._request_quote(session)

    def start(self, on_event: MarketEventHandler) -> None:
        with self._provider_lock:
            if self._worker is not None and self._worker.is_alive():
                if self._stop_event.is_set():
                    raise RuntimeError("Coinbase quote worker is still stopping")
                return
            self._set_event_handler(on_event)
            self._stop_event.clear()
            self._set_provider_state(
                ProviderState.CONNECTING, "Reading Coinbase public USDT/USD quotes"
            )
            self._worker = threading.Thread(
                target=self._run, name="coinbase-public-quotes", daemon=True
            )
            self._worker.start()

    def _publish_quote(self, event: MarketDataEvent) -> None:
        sequence = event.sequence
        assert sequence is not None  # normalize_message requires the upstream integer.
        with self._provider_lock:
            handler = self._provider_event_handler
            if self._stop_event.is_set() or handler is None:
                return
            if self._last_sequence is not None and sequence <= self._last_sequence:
                return
            if self._last_source_time is not None and event.event_ts < self._last_source_time:
                return
            self._last_sequence, self._last_source_time = sequence, event.event_ts
            self._provider_status = replace(
                self._provider_status,
                state=ProviderState.CONNECTED,
                message="Receiving Coinbase public USDT/USD quotes",
                last_event_ts=event.event_ts,
                last_receive_ts=event.receive_ts,
                events_received=self._provider_status.events_received + 1,
            )
        # Admission is fenced by stop under the lock; an admitted external
        # callback may finish after stop returns and must never hold that lock.
        try:
            handler(event)
        except Exception as exc:
            with self._provider_lock:
                if not self._stop_event.is_set():
                    self._set_provider_error(f"Market-data event handler failed: {exc}")

    def _run(self) -> None:
        session = self._session if self._session is not None else requests.Session()
        if self._session is None:
            session.trust_env = False
        try:
            while not self._stop_event.is_set():
                event = self._request_quote(session)
                self._publish_quote(event)
                if self.status.state == ProviderState.ERROR:
                    return
                self._stop_event.wait(self.poll_seconds)
        except (requests.RequestException, ValueError, TypeError, InterruptedError) as exc:
            with self._provider_lock:
                if not self._stop_event.is_set():
                    self._set_provider_error(
                        f"Coinbase public quote failed: {type(exc).__name__}: {exc}"
                    )
        finally:
            if self._session is None:
                session.close()

    def stop(self) -> None:
        """Fence new callback admissions; an admitted callback may finish later."""
        with self._provider_lock:
            self._stop_event.set()
            self._provider_event_handler = None
            self._set_provider_state(ProviderState.STOPPED, "Coinbase public quotes stopped")
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)
