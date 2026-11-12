"""Read-only OANDA v20 pricing-stream adapter for spot FX quotes.

The adapter uses only ``GET /v3/accounts/{accountID}/pricing`` and
``GET /v3/accounts/{accountID}/pricing/stream``. It never invokes OANDA order,
transaction, account-management, or execution endpoints. OANDA prices are
account-specific, quote-only observations; the emitted canonical book-shaped
snapshot carries bid/ask liquidity levels but must not be treated as a
centralized exchange order book or as public trade prints.

The v20 API requires an account ID and personal access token for the pricing
routes. Set ``OANDA_ACCESS_TOKEN`` or inject ``access_token``. The adapter
uses the documented practice or live streaming host, consumes the provider's
newline-delimited JSON messages and heartbeat records, and does not poll or
throttle the provider's price stream (which itself sends at most four prices
per second per instrument).
"""

from datetime import datetime, timezone
import json
import math
import os
import re
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import quote

import requests

from src.market.events import MarketDataEvent, MarketEventKind
from src.market.instruments import AssetClass, InstrumentSpec

from .market_data import ProviderCapabilities, ProviderState, ProviderStatusMixin


_ENVIRONMENT_URLS = {
    "practice": ("https://api-fxpractice.oanda.com", "https://stream-fxpractice.oanda.com"),
    "live": ("https://api-fxtrade.oanda.com", "https://stream-fxtrade.oanda.com"),
}


class OandaPricingProvider(ProviderStatusMixin):
    """Normalize OANDA v20 account prices as quote-only FX events.

    Args:
        account_id: OANDA v20 account identifier used only in pricing URLs.
        instrument_map: Explicit mapping from OANDA names (for example
            ``EUR_USD``) to canonical FX ``InstrumentSpec`` values.
        environment: ``"practice"`` or ``"live"``. Defaults to practice.
        access_token: Optional personal access token; otherwise reads
            ``OANDA_ACCESS_TOKEN`` when a request is started.
        session: Optional requests-compatible session for testing or custom
            transport. Only its ``get`` method is used.
        connect_timeout: HTTP connection timeout in seconds.
        read_timeout: Stream inactivity timeout. OANDA heartbeats are sent
            every five seconds, so this must exceed that interval.
        reconnect: Whether to reconnect the stream with bounded backoff.
        reconnect_max_delay: Maximum retry delay in seconds.
        receive_clock: Injectable local receive-time clock in epoch seconds.
    """

    name = "oanda"
    MAX_PRICE_UPDATES_PER_SECOND = 4
    HEARTBEAT_INTERVAL_SECONDS = 5

    def __init__(
        self,
        account_id: str,
        instrument_map: Mapping[str, InstrumentSpec],
        environment: str = "practice",
        access_token: Optional[str] = None,
        session: Any = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 20.0,
        reconnect: bool = True,
        reconnect_max_delay: float = 30.0,
        receive_clock: Callable[[], float] = time.time,
    ) -> None:
        ProviderStatusMixin.__init__(self)
        if not isinstance(account_id, str) or not account_id.strip():
            raise ValueError("account_id must be a non-empty OANDA v20 account identifier")
        if not instrument_map:
            raise ValueError("instrument_map must explicitly map OANDA instrument names to InstrumentSpec values")
        env = str(environment).strip().lower()
        if env not in _ENVIRONMENT_URLS:
            raise ValueError("environment must be 'practice' or 'live'")
        for name, value in (("connect_timeout", connect_timeout), ("read_timeout", read_timeout),
                            ("reconnect_max_delay", reconnect_max_delay)):
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError("{} must be a finite positive number".format(name))
            if not math.isfinite(numeric) or numeric <= 0:
                raise ValueError("{} must be a finite positive number".format(name))
            if name == "connect_timeout":
                connect_timeout = numeric
            elif name == "read_timeout":
                read_timeout = numeric
            else:
                reconnect_max_delay = numeric
        if read_timeout <= self.HEARTBEAT_INTERVAL_SECONDS:
            raise ValueError("read_timeout must exceed OANDA's five-second pricing heartbeat interval")
        if not callable(receive_clock):
            raise TypeError("receive_clock must be callable")

        normalized: Dict[str, InstrumentSpec] = {}
        for external_name, spec in instrument_map.items():
            if not isinstance(external_name, str) or not external_name.strip():
                raise ValueError("OANDA instrument names must be non-empty strings")
            key = external_name.strip().upper()
            if key in normalized:
                raise ValueError("instrument_map contains duplicate normalized OANDA instrument {!r}".format(key))
            if not isinstance(spec, InstrumentSpec):
                raise TypeError("instrument_map values must be InstrumentSpec instances")
            if spec.asset_class != AssetClass.FX:
                raise ValueError("OANDA pricing adapter accepts FX InstrumentSpec values only")
            normalized[key] = spec

        rest_base, stream_base = _ENVIRONMENT_URLS[env]
        self.account_id = account_id.strip()
        self.instrument_map = normalized
        self.environment = env
        self.rest_base_url = rest_base
        self.stream_base_url = stream_base
        self.access_token = access_token
        self.session = session if session is not None else requests.Session()
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.reconnect = bool(reconnect)
        self.reconnect_max_delay = reconnect_max_delay
        self.receive_clock = receive_clock
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._response = None
        self._stream_lock = threading.RLock()

    @property
    def capabilities(self) -> ProviderCapabilities:
        """OANDA pricing is quote-only; it provides neither trades nor a central book."""

        return ProviderCapabilities(quote=True, order_book=False, trades=False, sessions=False, status=True)

    def _token(self) -> str:
        token = self.access_token or os.environ.get("OANDA_ACCESS_TOKEN", "")
        if not token.strip():
            raise RuntimeError(
                "OANDA pricing requires OANDA_ACCESS_TOKEN or access_token plus an account_id; "
                "the provider calls pricing GET routes only."
            )
        return token.strip()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": "Bearer {}".format(self._token()),
            "Accept": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }

    def _instrument_query(self) -> str:
        return ",".join(self.instrument_map.keys())

    @staticmethod
    def _parse_timestamp(value: Any) -> float:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("OANDA price messages require an RFC3339 time")
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        # OANDA commonly emits nanosecond RFC3339 timestamps, while
        # datetime.fromisoformat accepts at most six fractional digits.
        text = re.sub(r"(\.\d{6})\d+(?=[+-]\d{2}:\d{2}$)", r"\1", text)
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("invalid OANDA RFC3339 timestamp {!r}".format(value)) from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("OANDA timestamp must include a timezone")
        timestamp = parsed.astimezone(timezone.utc).timestamp()
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("OANDA timestamp must be a finite epoch time")
        return timestamp

    @staticmethod
    def _parse_levels(raw_levels: Any, side_name: str) -> Tuple[Tuple[float, float], ...]:
        if raw_levels is None:
            return ()
        if not isinstance(raw_levels, (list, tuple)):
            raise ValueError("OANDA {} levels must be a list".format(side_name))
        levels = []
        for level in raw_levels:
            if not isinstance(level, Mapping):
                continue
            try:
                price = float(level["price"])
                liquidity = float(level["liquidity"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(price) or price <= 0 or not math.isfinite(liquidity) or liquidity <= 0:
                continue
            levels.append((price, liquidity))
        levels.sort(key=lambda item: item[0], reverse=(side_name == "bids"))
        return tuple(levels)

    def normalize_message(self, message: Mapping[str, Any], receive_ts: Optional[float] = None
                          ) -> Tuple[MarketDataEvent, ...]:
        """Normalize one decoded OANDA pricing-stream JSON line.

        Heartbeats update liveness only. A validated trading status precedes
        its quote, so a halted quote cannot fill orders before the halt applies.
        """
        if not isinstance(message, Mapping):
            raise TypeError("OANDA stream messages must be JSON objects")
        now = float(self.receive_clock() if receive_ts is None else receive_ts)
        kind = str(message.get("type", "")).upper()
        if kind == "HEARTBEAT":
            # Liveness is local receipt time; source time remains available
            # only for market events and should not be mistaken for freshness.
            self._parse_timestamp(message.get("time"))  # Validate the source heartbeat timestamp.
            self._set_heartbeat(now)
            return ()
        if kind != "PRICE":
            # The pricing stream may contain future informational message
            # types. Ignore them without misclassifying them as fills.
            return ()

        external_name = str(message.get("instrument", "")).strip().upper()
        spec = self.instrument_map.get(external_name)
        if spec is None:
            raise ValueError("unmapped OANDA instrument {!r}; add it to instrument_map".format(external_name))
        event_ts = self._parse_timestamp(message.get("time"))
        bids = self._parse_levels(message.get("bids", []), "bids")
        asks = self._parse_levels(message.get("asks", []), "asks")
        events = [MarketDataEvent(
            instrument_id=spec.instrument_id,
            source=self.name,
            kind=MarketEventKind.BOOK_SNAPSHOT,
            event_ts=event_ts,
            receive_ts=now,
            bids=bids,
            asks=asks,
        )]
        if message.get("tradeable") is not None:
            raw_tradeable = message["tradeable"]
            if isinstance(raw_tradeable, bool):
                tradeable = raw_tradeable
            elif isinstance(raw_tradeable, str) and raw_tradeable.strip().lower() in ("true", "false"):
                tradeable = raw_tradeable.strip().lower() == "true"
            else:
                raise ValueError("OANDA tradeable field must be a boolean")
            events.insert(0, MarketDataEvent(
                instrument_id=spec.instrument_id,
                source=self.name,
                kind=MarketEventKind.STATUS,
                event_ts=event_ts,
                receive_ts=now,
                status="tradeable" if tradeable else "non_tradeable",
            ))
        return tuple(events)

    def _handle_message(self, message: Mapping[str, Any]) -> None:
        try:
            events = self.normalize_message(message)
            if str(message.get("type", "")).upper() == "HEARTBEAT":
                if self.status.state != ProviderState.CONNECTED:
                    self._set_provider_state(ProviderState.CONNECTED, "Receiving OANDA pricing heartbeat")
                return
            if events:
                self._set_provider_state(ProviderState.CONNECTED, "Receiving OANDA account prices")
                for event in events:
                    self._publish(event)
        except Exception as exc:
            self._set_provider_error("Unable to normalize OANDA pricing message: {}".format(exc))

    def fetch_prices(self) -> Tuple[MarketDataEvent, ...]:
        """Fetch a one-shot public-market pricing snapshot through OANDA's GET route."""
        account_path = quote(self.account_id, safe="")
        url = "{}/v3/accounts/{}/pricing".format(self.rest_base_url, account_path)
        response = None
        try:
            response = self.session.get(
                url,
                params={"instruments": self._instrument_query()},
                headers=self._headers(),
                timeout=self.connect_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError("OANDA pricing response must be a JSON object")
            events = []
            now = float(self.receive_clock())
            for price in payload.get("prices", []):
                events.extend(self.normalize_message(price, receive_ts=now))
            self._set_provider_state(ProviderState.STOPPED, "Fetched OANDA pricing snapshot")
            for event in events:
                self._publish(event)
            return tuple(events)
        except Exception as exc:
            self._set_provider_error("OANDA pricing GET failed: {}".format(exc))
            raise
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass

    def start(self, on_event: Callable[[MarketDataEvent], None]) -> None:
        """Start the OANDA pricing stream in a managed background worker."""
        self._set_event_handler(on_event)
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            self._token()  # Fail immediately with an actionable configuration error.
        except Exception as exc:
            self._set_provider_error(str(exc))
            raise
        self._stop_event.clear()
        self._set_provider_state(ProviderState.CONNECTING, "Opening OANDA pricing stream")
        self._worker = threading.Thread(target=self._stream_loop, name="oanda-pricing-stream", daemon=True)
        self._worker.start()

    def _stream_loop(self) -> None:
        delay = 1.0
        while not self._stop_event.is_set():
            response = None
            try:
                account_path = quote(self.account_id, safe="")
                url = "{}/v3/accounts/{}/pricing/stream".format(self.stream_base_url, account_path)
                response = self.session.get(
                    url,
                    params={"instruments": self._instrument_query(), "snapshot": "true"},
                    headers=self._headers(),
                    stream=True,
                    timeout=(self.connect_timeout, self.read_timeout),
                )
                response.raise_for_status()
                with self._stream_lock:
                    self._response = response
                self._set_provider_state(ProviderState.CONNECTED, "OANDA pricing stream connected")
                delay = 1.0
                for line in response.iter_lines(decode_unicode=True):
                    if self._stop_event.is_set():
                        break
                    if line is None or line == b"" or line == "":
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="strict")
                    message = json.loads(line)
                    if isinstance(message, Mapping):
                        self._handle_message(message)
                if self._stop_event.is_set():
                    break
                raise ConnectionError("OANDA pricing stream ended")
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._set_provider_error("OANDA pricing stream error: {}".format(exc))
                if not self.reconnect:
                    break
                if self._stop_event.wait(delay):
                    break
                delay = min(delay * 2.0, self.reconnect_max_delay)
                self._set_provider_state(ProviderState.CONNECTING, "Retrying OANDA pricing stream")
            finally:
                with self._stream_lock:
                    if self._response is response:
                        self._response = None
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass
        if self._stop_event.is_set():
            self._set_provider_state(ProviderState.STOPPED, "OANDA pricing stream stopped")

    def stop(self) -> None:
        """Stop the quote stream and close the active HTTP response."""
        self._stop_event.set()
        with self._stream_lock:
            response = self._response
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=self.connect_timeout + 1.0)
        if worker is None or not worker.is_alive():
            self._set_provider_state(ProviderState.STOPPED, "OANDA pricing stream stopped")

