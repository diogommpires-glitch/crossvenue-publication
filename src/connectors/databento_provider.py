"""Optional read-only Databento live market-data adapter.

The adapter subscribes only to market-data schemas. It never creates or
modifies orders. Callers must provide an explicit raw-symbol to
``InstrumentSpec`` mapping for the selected dataset; numeric Databento
instrument identifiers are learned from definition records or an optional
explicit ID mapping.

Install the optional SDK with ``pip install databento``. Tests can inject a
small client object and exercise record normalization without the SDK, an API
key, or a network connection.
"""

import importlib
import math
import time
from numbers import Integral, Real
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from src.market.events import MarketDataEvent, MarketEventKind
from src.market.instruments import AssetClass, InstrumentSpec
from src.utils.types import Side

from .market_data import ProviderCapabilities, ProviderState, ProviderStatus, ProviderStatusMixin


_PRICE_SCALE = 1_000_000_000.0
_UNDEFINED_PRICE = -(2 ** 63)
_UNDEFINED_TIMESTAMP = (2 ** 64) - 1
_SUPPORTED_SCHEMAS = frozenset(("definition", "trades", "mbp-1", "mbp-10", "cmbp-1", "status"))


class DatabentoMarketDataProvider(ProviderStatusMixin):
    """Normalize selected Databento equity and futures records.

    Args:
        dataset: Explicit Databento dataset, for example ``XNAS.ITCH`` or
            ``GLBX.MDP3``. One provider instance subscribes to one dataset.
        symbol_map: Explicit mapping from provider raw symbol to canonical
            instrument metadata. Equities and futures are accepted.
        schemas: Databento record schemas. ``definition`` maps provider numeric
            instrument IDs back to the explicit ``symbol_map``. A ``trades``
            schema is preferred for trades; book schemas emit a trade as well
            only when no separate ``trades`` schema is configured.
        stype_in: Input symbology type used for the subscription.
        api_key: Optional Databento key. If omitted, the official SDK reads
            ``DATABENTO_API_KEY`` from the environment.
        client: Optional pre-created SDK-compatible client, mainly for tests.
        client_factory: Optional callable receiving ``key=...`` to construct
            an SDK-compatible client.
        instrument_id_map: Optional mapping from Databento numeric instrument
            ID to a raw symbol or canonical ``InstrumentSpec``.
        receive_clock: Injectable local receive-time clock, in epoch seconds.
    """

    name = "databento"

    def __init__(
        self,
        dataset: str,
        symbol_map: Mapping[str, InstrumentSpec],
        schemas: Sequence[str] = ("definition", "mbp-1", "trades"),
        stype_in: str = "raw_symbol",
        api_key: Optional[str] = None,
        client: Any = None,
        client_factory: Optional[Callable[..., Any]] = None,
        instrument_id_map: Optional[Mapping[int, Any]] = None,
        receive_clock: Callable[[], float] = time.time,
    ) -> None:
        ProviderStatusMixin.__init__(self)
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("dataset must be an explicit non-empty Databento dataset")
        if not isinstance(stype_in, str) or not stype_in.strip():
            raise ValueError("stype_in must be a non-empty symbology type")
        if not symbol_map:
            raise ValueError("symbol_map must explicitly map provider symbols to InstrumentSpec values")

        normalized_symbols: Dict[str, InstrumentSpec] = {}
        for symbol, spec in symbol_map.items():
            raw_symbol = self._normalize_symbol(symbol)
            if not isinstance(spec, InstrumentSpec):
                raise TypeError("symbol_map values must be InstrumentSpec instances")
            if spec.asset_class not in (AssetClass.EQUITY, AssetClass.FUTURE):
                raise ValueError("Databento adapter supports equity and futures instruments only")
            if raw_symbol in normalized_symbols:
                raise ValueError("symbol_map contains duplicate normalized symbol {!r}".format(raw_symbol))
            normalized_symbols[raw_symbol] = spec

        normalized_schemas = tuple(str(schema).strip().lower() for schema in schemas)
        if not normalized_schemas or any(not schema for schema in normalized_schemas):
            raise ValueError("schemas must contain at least one non-empty record schema")
        unknown_schemas = sorted(set(normalized_schemas) - _SUPPORTED_SCHEMAS)
        if unknown_schemas:
            raise ValueError(
                "unsupported Databento schema(s): {}. Supported: {}".format(
                    ", ".join(unknown_schemas), ", ".join(sorted(_SUPPORTED_SCHEMAS))
                )
            )
        if len(set(normalized_schemas)) != len(normalized_schemas):
            raise ValueError("schemas cannot contain duplicates")
        if not callable(receive_clock):
            raise TypeError("receive_clock must be callable")

        self.dataset = dataset.strip()
        self.stype_in = stype_in.strip()
        self.schemas = normalized_schemas
        self.api_key = api_key
        self.symbol_map = normalized_symbols
        self.receive_clock = receive_clock
        self._client = client
        self._client_factory = client_factory
        self._explicit_instrument_ids: Dict[int, InstrumentSpec] = {}
        for databento_id, mapped in (instrument_id_map or {}).items():
            try:
                numeric_id = int(databento_id)
            except (TypeError, ValueError, OverflowError):
                raise ValueError("instrument_id_map keys must be non-negative integers")
            if numeric_id < 0:
                raise ValueError("instrument_id_map keys must be non-negative integers")
            if isinstance(mapped, InstrumentSpec):
                spec = mapped
            elif isinstance(mapped, str):
                raw_symbol = self._normalize_symbol(mapped)
                spec = normalized_symbols.get(raw_symbol)
                if spec is None:
                    raise ValueError("instrument_id_map symbol {!r} is not in symbol_map".format(mapped))
            else:
                raise TypeError("instrument_id_map values must be raw symbols or InstrumentSpec instances")
            if spec not in normalized_symbols.values():
                raise ValueError("instrument_id_map instruments must also be present in symbol_map")
            self._explicit_instrument_ids[numeric_id] = spec

        self._instrument_ids: Dict[int, InstrumentSpec] = dict(self._explicit_instrument_ids)
        self._raw_symbols_by_instrument = {
            spec.instrument_id: raw_symbol for raw_symbol, spec in normalized_symbols.items()
        }
        self._subscribed = False
        self._callback_errors = 0

    @property
    def status(self) -> ProviderStatus:
        status = super().status
        connected = getattr(self._client, "is_connected", None)
        if (self._subscribed and status.state != ProviderState.ERROR
                and callable(connected) and not connected()):
            self._set_provider_error("Databento live connection closed")
        return super().status

    @property
    def capabilities(self) -> ProviderCapabilities:
        schemas = set(self.schemas)
        book = bool(schemas.intersection(("mbp-1", "mbp-10", "cmbp-1")))
        trades = "trades" in schemas or (book and "trades" not in schemas)
        return ProviderCapabilities(
            quote=book,
            order_book=book,
            trades=trades,
            sessions="status" in schemas,
            status=True,
        )

    @staticmethod
    def _normalize_symbol(value: Any) -> str:
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="strict")
        symbol = str(value).replace("\x00", "").strip().upper()
        if not symbol:
            raise ValueError("provider symbols must be non-empty")
        return symbol

    @staticmethod
    def _get(record: Any, name: str, default: Any = None) -> Any:
        if isinstance(record, Mapping):
            return record.get(name, default)
        return getattr(record, name, default)

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("ascii", errors="replace").replace("\x00", "").strip()
        name = getattr(value, "name", None)
        if name:
            return str(name).split(".")[-1].upper()
        raw_value = getattr(value, "value", value)
        if isinstance(raw_value, bytes):
            return raw_value.decode("ascii", errors="replace").replace("\x00", "").strip()
        return str(raw_value).replace("\x00", "").strip()

    @classmethod
    def _price(cls, raw: Any) -> Optional[float]:
        if raw is None:
            return None
        # Databento DBN prices are scaled integer values. Test fixtures and
        # injected adapters sometimes already provide normalized floats; do
        # not truncate those through int() before applying the DBN scale.
        if isinstance(raw, Real) and not isinstance(raw, Integral):
            floating = float(raw)
            return floating if math.isfinite(floating) else None
        try:
            numeric = int(raw)
        except (TypeError, ValueError, OverflowError):
            try:
                floating = float(raw)
            except (TypeError, ValueError, OverflowError):
                return None
            return floating if math.isfinite(floating) else None
        if numeric == _UNDEFINED_PRICE:
            return None
        result = numeric / _PRICE_SCALE
        return result if math.isfinite(result) else None

    @classmethod
    def _timestamp(cls, raw: Any, fallback: Optional[float] = None) -> Optional[float]:
        if raw is None:
            return fallback
        if isinstance(raw, Real) and not isinstance(raw, Integral):
            floating = float(raw)
            if not math.isfinite(floating) or floating < 0:
                return fallback
            # Non-integral values are normally already expressed in epoch
            # seconds by injected clients; DBN-native timestamps are integers
            # in nanoseconds.
            if floating < 1_000_000_000_000:
                return floating
            numeric = int(floating)
        else:
            try:
                numeric = int(raw)
            except (TypeError, ValueError, OverflowError):
                try:
                    numeric = int(float(raw))
                except (TypeError, ValueError, OverflowError):
                    return fallback
        if numeric == _UNDEFINED_TIMESTAMP or numeric < 0:
            return fallback
        result = numeric / 1_000_000_000.0
        return result if math.isfinite(result) else fallback

    @classmethod
    def _side(cls, raw: Any) -> Optional[Side]:
        side = cls._text(raw).upper()
        if side in ("B", "BID", "BUY"):
            return Side.BUY
        if side in ("A", "ASK", "SELL"):
            return Side.SELL
        return None

    def _schema_for(self, record: Any) -> str:
        explicit = self._get(record, "schema") or self._get(record, "schema_name")
        if explicit:
            return str(explicit).strip().lower().replace("_", "-")
        class_name = type(record).__name__.lower()
        rtype = self._get(record, "rtype")
        if class_name in {"errormsg", "errormsgv1"} or (rtype is not None and int(rtype) == 21):
            return "error"
        if "definition" in class_name or (rtype is not None and int(rtype) == 19):
            return "definition"
        if "trade" in class_name or (rtype is not None and int(rtype) == 0):
            return "trades"
        if "mbp10" in class_name:
            return "mbp-10"
        if "mbp1" in class_name or "cmbp1" in class_name:
            return "cmbp-1" if "cmbp1" in class_name else "mbp-1"
        if "status" in class_name or "stat" in class_name:
            return "status"
        if self._get(record, "levels") is not None or self._get(record, "bid_px_00") is not None:
            return "mbp-1"
        if self._get(record, "raw_symbol") is not None:
            return "definition"
        if self._get(record, "price") is not None and self._get(record, "size") is not None:
            action = self._text(self._get(record, "action")).upper()
            return "trades" if action in ("T", "TRADE") else ""
        return ""

    def _spec_for(self, record: Any) -> Optional[InstrumentSpec]:
        raw_symbol = self._get(record, "raw_symbol")
        if raw_symbol is not None:
            normalized = self._normalize_symbol(raw_symbol)
            spec = self.symbol_map.get(normalized)
            if spec is not None:
                try:
                    databento_id = int(self._get(record, "instrument_id"))
                except (TypeError, ValueError, OverflowError):
                    return spec
                self._instrument_ids[databento_id] = spec
                return spec

        raw_id = self._get(record, "instrument_id")
        try:
            databento_id = int(raw_id)
        except (TypeError, ValueError, OverflowError):
            return None
        return self._instrument_ids.get(databento_id)

    def _event_times(self, record: Any, receive_ts: float) -> Tuple[float, float]:
        event_ts = self._timestamp(self._get(record, "ts_event"), fallback=receive_ts)
        # receive_ts is the adapter's local clock; DBN ts_recv is the vendor's
        # gateway receipt time and is not interchangeable with local receipt.
        return (receive_ts if event_ts is None else event_ts, receive_ts)

    def _book_levels(self, record: Any) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[Tuple[float, float], ...]]:
        bids = []
        asks = []
        levels = self._get(record, "levels")
        if levels is not None:
            try:
                source = list(levels)
            except TypeError:
                source = []
            for level in source:
                bid_price = self._price(self._get(level, "bid_px", self._get(level, "bid_price")))
                ask_price = self._price(self._get(level, "ask_px", self._get(level, "ask_price")))
                bid_size = self._get(level, "bid_sz", self._get(level, "bid_size", 0))
                ask_size = self._get(level, "ask_sz", self._get(level, "ask_size", 0))
                try:
                    bid_size = float(bid_size)
                    ask_size = float(ask_size)
                except (TypeError, ValueError, OverflowError):
                    continue
                if bid_price is not None and math.isfinite(bid_size) and bid_size > 0:
                    bids.append((bid_price, bid_size))
                if ask_price is not None and math.isfinite(ask_size) and ask_size > 0:
                    asks.append((ask_price, ask_size))
        else:
            for index in range(10):
                suffix = "_{:02d}".format(index)
                bid_price = self._price(self._get(record, "bid_px" + suffix))
                ask_price = self._price(self._get(record, "ask_px" + suffix))
                bid_size = self._get(record, "bid_sz" + suffix, 0)
                ask_size = self._get(record, "ask_sz" + suffix, 0)
                try:
                    bid_size = float(bid_size)
                    ask_size = float(ask_size)
                except (TypeError, ValueError, OverflowError):
                    continue
                if bid_price is not None and math.isfinite(bid_size) and bid_size > 0:
                    bids.append((bid_price, bid_size))
                if ask_price is not None and math.isfinite(ask_size) and ask_size > 0:
                    asks.append((ask_price, ask_size))
        bids.sort(key=lambda level: level[0], reverse=True)
        asks.sort(key=lambda level: level[0])
        return tuple(bids), tuple(asks)

    def normalize_record(self, record: Any, receive_ts: Optional[float] = None) -> Tuple[MarketDataEvent, ...]:
        """Normalize one injected/provider DBN record; definitions update the symbol map."""
        now = float(self.receive_clock() if receive_ts is None else receive_ts)
        schema = self._schema_for(record)
        if schema == "definition":
            # Definitions are reference data used only to bind the numeric ID
            # to the explicit caller-supplied raw symbol mapping.
            self._spec_for(record)
            return ()

        spec = self._spec_for(record)
        if spec is None:
            raw_id = self._get(record, "instrument_id", "unknown")
            raw_symbol = self._get(record, "raw_symbol", "unknown")
            self._set_provider_error(
                "Unmapped Databento instrument {!r}/{!r}; provide an instrument_id_map "
                "or subscribe to definition records matching symbol_map".format(raw_id, raw_symbol)
            )
            return ()

        event_ts, local_receive_ts = self._event_times(record, now)
        sequence = self._get(record, "sequence")
        if sequence is not None:
            try:
                sequence = int(sequence)
            except (TypeError, ValueError, OverflowError):
                sequence = None
        base = {
            "instrument_id": spec.instrument_id,
            "source": self.name,
            "event_ts": event_ts,
            "receive_ts": local_receive_ts,
            "sequence": sequence,
        }

        if schema == "status":
            missing = object()
            trading = self._get(record, "is_trading", missing)
            raw_status = self._text(self._get(record, "status", "")).strip().lower()
            action = self._text(self._get(record, "action", "")).strip().lower()
            if isinstance(trading, bool):
                status = "trading" if trading else "closed"
            elif trading is not missing:
                trading_text = self._text(trading).strip().lower()
                if trading_text in {"y", "yes", "true", "1", "t"}:
                    status = "trading"
                elif trading_text in {"n", "no", "false", "0", "f"}:
                    status = "closed"
                else:
                    status = "unknown"
            elif raw_status in {"trading", "open", "active", "tradeable", "continuous"}:
                status = "trading"
            elif raw_status in {"halt", "halted", "closed", "paused", "non_tradeable"}:
                status = "halted"
            elif action in {"7", "t", "trade", "trading"}:
                status = "trading"
            elif action in {"8", "h", "halt", "halted"}:
                status = "halted"
            else:
                status = raw_status or action or "unknown"
            session = self._text(self._get(record, "session", "")) or None
            return (MarketDataEvent(
                **base,
                kind=MarketEventKind.STATUS,
                status=status,
                session=session,
            ),)

        if schema == "trades":
            price = self._price(self._get(record, "price"))
            raw_quantity = self._get(record, "size", self._get(record, "quantity"))
            try:
                quantity = float(raw_quantity)
            except (TypeError, ValueError, OverflowError):
                quantity = float("nan")
            if price is None or not math.isfinite(quantity) or quantity <= 0:
                return ()
            return (MarketDataEvent(
                **base,
                kind=MarketEventKind.TRADE,
                price=price,
                quantity=quantity,
                side=self._side(self._get(record, "side")),
            ),)

        if schema in ("mbp-1", "mbp-10", "cmbp-1"):
            bids, asks = self._book_levels(record)
            events = [MarketDataEvent(
                **base,
                kind=MarketEventKind.BOOK_SNAPSHOT,
                bids=bids,
                asks=asks,
            )]
            action = self._text(self._get(record, "action")).upper()
            if "trades" not in self.schemas and action in ("T", "TRADE"):
                price = self._price(self._get(record, "price"))
                raw_quantity = self._get(record, "size", self._get(record, "quantity"))
                try:
                    quantity = float(raw_quantity)
                except (TypeError, ValueError, OverflowError):
                    quantity = 0.0
                if price is not None and math.isfinite(quantity) and quantity > 0:
                    events.append(MarketDataEvent(
                        **base,
                        kind=MarketEventKind.TRADE,
                        price=price,
                        quantity=quantity,
                        side=self._side(self._get(record, "side")),
                    ))
            return tuple(events)

        return ()

    def handle_record(self, record: Any) -> None:
        """SDK callback: normalize and forward records without raising into the SDK."""
        try:
            if self._schema_for(record) == "error":
                self._set_provider_error("Databento gateway error: " + self._text(self._get(record, "err")))
                return
            if self.status.state == ProviderState.ERROR:
                return
            events = self.normalize_record(record)
            if events:
                self._set_provider_state(ProviderState.CONNECTED, "Receiving Databento data")
                for event in events:
                    self._publish(event)
            elif self._schema_for(record) == "definition":
                self._set_provider_state(ProviderState.CONNECTED, "Receiving Databento definitions")
        except Exception as exc:
            self._callback_errors += 1
            self._set_provider_error("Unable to normalize Databento record: {}".format(exc))

    def start(self, on_event: Callable[[MarketDataEvent], None]) -> None:
        """Subscribe to public Databento schemas and start the injected/optional client."""
        self._set_event_handler(on_event)
        if self._subscribed:
            return
        self._set_provider_state(ProviderState.CONNECTING, "Creating Databento live subscription")
        try:
            if self._client is None:
                factory = self._client_factory
                if factory is None:
                    try:
                        databento = importlib.import_module("databento")
                    except ImportError as exc:
                        raise ImportError(
                            "Databento support is optional. Install it with `pip install databento` "
                            "and set DATABENTO_API_KEY, or inject a compatible client."
                        ) from exc
                    factory = databento.Live
                self._client = factory(key=self.api_key) if self.api_key else factory()

            for schema in self.schemas:
                self._client.subscribe(
                    dataset=self.dataset,
                    schema=schema,
                    stype_in=self.stype_in,
                    symbols=list(self.symbol_map.keys()),
                )
            self._client.add_callback(self.handle_record)
            self._client.start()
            self._subscribed = True
            # Some client implementations deliver buffered records during
            # start(); preserve CONNECTED if that synchronous callback already
            # observed data.
            if self.status.state == ProviderState.CONNECTING:
                self._set_provider_state(ProviderState.CONNECTING, "Subscribed; waiting for the first Databento record")
        except Exception as exc:
            self._set_provider_error("Databento subscription failed: {}".format(exc))
            raise

    def stop(self) -> None:
        """Stop the Databento client and mark the read-only feed stopped."""
        client = self._client
        self._subscribed = False
        if client is not None:
            try:
                client.stop()
            except Exception as exc:
                self._set_provider_error("Databento client stop failed: {}".format(exc))
                return
        self._set_provider_state(ProviderState.STOPPED, "Databento feed stopped")

