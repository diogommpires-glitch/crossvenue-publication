"""Independent supervision for read-only market-data connections."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import random
import re
import threading
import time
from typing import Any, Callable, Optional


def safe_provider_error(error: Any) -> str:
    """Keep actionable errors without retaining tokens or account URLs."""
    value = str(error)
    value = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[redacted]", value)
    value = re.sub(r"(?i)(api[_-]?key|token|authorization|password)(\s*[=:]\s*)[^\s,;]+",
                   r"\1\2[redacted]", value)
    value = re.sub(r"(?i)(/accounts/)[^/\s?]+", r"\1[redacted]", value)
    return value[:500]


def is_access_error(error: Any) -> bool:
    message = str(error).lower()
    return any(word in message for word in (
        "401", "403", "unauthorized", "forbidden", "entitlement", "permission denied",
        "access denied", "api key", "api_key", "access token", "credential", "needs account id",
        "account_id is required", "account id in environment"))


@dataclass(frozen=True)
class SupervisorStatus:
    provider_id: str
    instrument_ids: tuple
    state: str = "stopped"
    generation: int = 0
    message: str = ""
    attempts: int = 0
    next_retry_at: Optional[float] = None
    last_event_at: Optional[float] = None


class ProviderSupervisor:
    """One worker per provider; an error cannot roll back another provider."""

    def __init__(self, provider_id: str, instrument_ids: tuple, factory: Callable,
                 on_event: Callable, on_bind: Callable, on_invalidate: Callable,
                 *, clock=time.time, random_source=None, retry_cap: float = 60.0,
                 poll_seconds: float = 0.2, on_status=None):
        self.provider_id = provider_id
        self.instrument_ids = tuple(instrument_ids)
        self.factory = factory
        self.on_event = on_event
        self.on_bind = on_bind
        self.on_invalidate = on_invalidate
        self.on_status = on_status or (lambda status: None)
        self.clock = clock
        self.random = random_source or random.Random()
        self.retry_cap = min(60.0, max(0.01, float(retry_cap)))
        self.poll_seconds = max(0.01, float(poll_seconds))
        self._lock = threading.RLock()
        self._status = SupervisorStatus(provider_id, self.instrument_ids)
        self._stop = threading.Event()
        self._restart = threading.Event()
        self._worker = None
        self._provider = None

    def _update(self, **changes):
        with self._lock:
            previous = self._status
            values = asdict(self._status)
            values.update(changes)
            values["instrument_ids"] = tuple(values["instrument_ids"])
            self._status = SupervisorStatus(**values)
            changed = any(getattr(previous, field) != getattr(self._status, field)
                          for field in ("state", "generation", "message"))
            status = asdict(self._status)
        if changed:
            self.on_status(status)

    @property
    def status(self):
        with self._lock:
            return asdict(self._status)

    def start(self):
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._restart.clear()
        self._worker = threading.Thread(target=self._run, name="feed-{}".format(self.provider_id), daemon=True)
        self._worker.start()

    def request_resync(self):
        """Invalidate this connection and obtain a new generation/snapshot."""
        self._restart.set()

    def retry_access(self):
        """An explicit local retry after correcting credentials/entitlements."""
        self._restart.set()

    def _run(self):
        delay = 1.0
        # Status reserves the next generation during backoff; each actual
        # connection attempt advances this counter only once.
        generation = self.status["generation"]
        while not self._stop.is_set():
            generation += 1
            self._update(state="connecting", generation=generation,
                         attempts=self.status["attempts"] + 1, next_retry_at=None,
                         last_event_at=None)
            self._restart.clear()
            provider = None
            try:
                provider = self.factory()
                self._provider = provider
                self.on_bind(self.provider_id, provider, self.instrument_ids)
                connection_valid = threading.Event()
                connection_valid.set()
                connected_sockets: set[str] = set()

                def sockets_connected(current_provider: Any = provider,
                                      established: set[str] = connected_sockets) -> bool:
                    snapshot = getattr(current_provider, "status_snapshot", None)
                    if not callable(snapshot):
                        return True
                    flags = snapshot()
                    # Initially false flags are normal while sockets connect.
                    # Once connected, losing either stream invalidates both.
                    for name in ("depth_connected", "trades_connected"):
                        if flags.get(name) is True:
                            established.add(name)
                    return all(flags.get(name) is not False for name in established)

                def deliver(event, current_generation=generation, valid=connection_valid,
                            check_sockets=sockets_connected):
                    if (self._stop.is_set() or not valid.is_set()
                            or current_generation != self.status["generation"]):
                        return
                    if not check_sockets():
                        valid.clear()
                        self._restart.set()
                        return
                    self._update(state="healthy", message="", last_event_at=float(self.clock()))
                    self.on_event(self.provider_id, event, current_generation)

                provider.start(deliver)
                if self.status["last_event_at"] is None:
                    self._update(state="connected", message="Waiting for market events")
                delay = 1.0
                while not self._stop.wait(self.poll_seconds):
                    if self._restart.is_set():
                        raise ConnectionError("Connection resynchronization requested")
                    if not sockets_connected():
                        connection_valid.clear()
                        raise ConnectionError("Previously connected market-data socket disconnected")
                    status = provider.status
                    state = str(getattr(status.state, "value", status.state)).lower()
                    if state == "error":
                        raise ConnectionError(status.message or "Provider connection failed")
                    if state == "stopped":
                        raise ConnectionError("Provider stopped unexpectedly")
                break
            except Exception as exc:
                if self._stop.is_set():
                    break
                message = safe_provider_error(exc)
                # Fence the failed callback generation throughout backoff,
                # while telling the writer which old generation was revoked.
                self._update(state="invalidated", generation=generation + 1, message=message)
                self.on_invalidate(self.provider_id, message, generation)
                configuration_error = isinstance(exc, (ValueError, ImportError)) or any(
                    word in str(exc).lower() for word in ("not installed", "optional sdk", "no module named"))
                if is_access_error(exc) or configuration_error:
                    self._update(state="access_denied" if is_access_error(exc) else "configuration_error",
                                 message=message, next_retry_at=None)
                    while not self._stop.wait(self.poll_seconds) and not self._restart.is_set():
                        pass
                    delay = 1.0
                else:
                    wait = min(self.retry_cap, delay * self.random.uniform(0.8, 1.2))
                    self._update(state="retrying", message=message,
                                 next_retry_at=float(self.clock()) + wait)
                    deadline = time.monotonic() + wait
                    while not self._stop.is_set() and not self._restart.is_set() and time.monotonic() < deadline:
                        self._stop.wait(min(self.poll_seconds, max(0, deadline - time.monotonic())))
                    delay = min(self.retry_cap, delay * 2)
            finally:
                if provider is not None:
                    try:
                        provider.stop()
                    except Exception:
                        pass
                self._provider = None
        self._update(state="stopped", next_retry_at=None)

    def stop(self):
        self._stop.set()
        self._restart.set()
        provider = self._provider
        if provider is not None:
            try:
                provider.stop()
            except Exception:
                pass
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=5.0)
