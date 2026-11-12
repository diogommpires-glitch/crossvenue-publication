"""Read-only provider contracts for normalized public market data.

Market-data providers do not expose account, order, or cancellation methods.
They publish immutable :class:`MarketDataEvent` values for configured
:class:`InstrumentSpec` instances and report feed health independently.
"""

from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Callable, Optional, Protocol, runtime_checkable

from src.market.events import MarketDataEvent


MarketEventHandler = Callable[[MarketDataEvent], None]


class ProviderState(str, Enum):
    """Lifecycle states shared by read-only provider adapters."""

    STOPPED = "stopped"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass(frozen=True)
class ProviderCapabilities:
    """Market-data features provided by one configured adapter instance.

    ``quote`` means the adapter can publish a two-sided or one-sided quote
    snapshot. It does not imply that the source is a centralized limit book.
    OANDA's account-specific FX prices, for example, are quote-only.
    """

    quote: bool = False
    order_book: bool = False
    trades: bool = False
    sessions: bool = False
    status: bool = True


@dataclass(frozen=True)
class ProviderStatus:
    """Observable connection and freshness state for a provider."""

    provider: str
    state: ProviderState
    message: str = ""
    last_event_ts: Optional[float] = None
    last_receive_ts: Optional[float] = None
    last_heartbeat_ts: Optional[float] = None
    events_received: int = 0

    @property
    def connected(self) -> bool:
        """Whether the provider currently has an active data connection."""

        return self.state == ProviderState.CONNECTED


@runtime_checkable
class MarketDataProvider(Protocol):
    """Minimal read-only provider interface consumed by application wiring."""

    @property
    def name(self) -> str:
        """Stable provider identifier used in normalized event provenance."""

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Capabilities for the configured subscriptions."""

    @property
    def status(self) -> ProviderStatus:
        """Current provider lifecycle and last-data state."""

    def start(self, on_event: MarketEventHandler) -> None:
        """Start reading the configured instruments and publish normalized events."""

    def stop(self) -> None:
        """Close the public data connection and release worker resources."""


class ProviderStatusMixin:
    """Thread-safe status and event delivery helpers for provider adapters."""

    name = "market-data"

    def __init__(self) -> None:
        self._provider_lock = threading.RLock()
        self._provider_event_handler: Optional[MarketEventHandler] = None
        self._provider_status = ProviderStatus(
            provider=self.name,
            state=ProviderState.STOPPED,
        )

    @property
    def status(self) -> ProviderStatus:
        with self._provider_lock:
            return self._provider_status

    def _set_provider_state(self, state: ProviderState, message: str = "") -> None:
        with self._provider_lock:
            current = self._provider_status
            self._provider_status = ProviderStatus(
                provider=self.name,
                state=state,
                message=message,
                last_event_ts=current.last_event_ts,
                last_receive_ts=current.last_receive_ts,
                last_heartbeat_ts=current.last_heartbeat_ts,
                events_received=current.events_received,
            )

    def _set_provider_error(self, message: str) -> None:
        self._set_provider_state(ProviderState.ERROR, message)

    def _set_heartbeat(self, timestamp: Optional[float] = None) -> None:
        heartbeat_time = time.time() if timestamp is None else float(timestamp)
        with self._provider_lock:
            current = self._provider_status
            self._provider_status = ProviderStatus(
                provider=self.name,
                state=current.state,
                message=current.message,
                last_event_ts=current.last_event_ts,
                last_receive_ts=current.last_receive_ts,
                last_heartbeat_ts=heartbeat_time,
                events_received=current.events_received,
            )

    def _publish(self, event: MarketDataEvent) -> None:
        handler = self._provider_event_handler
        if handler is None:
            return
        with self._provider_lock:
            current = self._provider_status
            self._provider_status = ProviderStatus(
                provider=self.name,
                state=current.state,
                message=current.message,
                last_event_ts=event.event_ts,
                last_receive_ts=event.receive_ts,
                last_heartbeat_ts=current.last_heartbeat_ts,
                events_received=current.events_received + 1,
            )
        try:
            handler(event)
        except Exception as exc:
            self._set_provider_error("Market-data event handler failed: {}".format(exc))

    def _set_event_handler(self, on_event: MarketEventHandler) -> None:
        if not callable(on_event):
            raise TypeError("on_event must be callable")
        self._provider_event_handler = on_event
