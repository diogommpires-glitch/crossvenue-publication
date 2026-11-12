"""Transport-independent, single-writer application boundary for paper accounts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import nullcontext
from typing import Any


class PaperApplicationService:
    """Serialize inputs and commands; publish cached revisioned read models.

    Provider threads never mutate the account. The small emergency channel
    permits Stop and stream invalidation even when market input is saturated.
    Incomplete generations are discarded explicitly and must resynchronize.
    """

    PROTOCOL_VERSION = 1
    READ_COMMANDS = frozenset({"get_status", "status_request", "get_trades", "get_history",
                               "get_diagnostics", "preview_migration", "get_account_actions"})

    def __init__(self, engine, state_manager=None, *, queue_capacity=10000,
                 recorder=None, analytics=None, research=None, clock=time.time):
        if not isinstance(queue_capacity, int) or queue_capacity < 1:
            raise ValueError("queue_capacity must be a positive integer")
        self.engine = engine
        self.engine.defer_dashboard_updates = True
        self.state_manager = state_manager or getattr(engine, "state_manager", None)
        self.account_id = engine.account_id
        self.clock = clock
        self._writer_time = float(clock())
        # Freeze time for each authoritative input so recordings reproduce all
        # decisions, rather than depending on sub-millisecond call timing.
        self.engine.clock = lambda: self._writer_time
        if getattr(self.engine, "ledger", None) is not None:
            self.engine.ledger.clock = self.engine.clock
        self.recorder = recorder
        recovery_halt = getattr(self.state_manager, "get_command_persistence_halt", None)
        reason = recovery_halt(self.account_id) if callable(recovery_halt) else None
        if reason is not None:
            self.engine.circuit_breaker_active = True
            self.engine.circuit_breaker_reason = reason
            self.engine.is_running = False
            self.engine._cancel_all_quotes(reason=reason)
            if self.recorder is not None:
                self.recorder.set_initial_state(self.engine._checkpoint_payload())
        self.analytics = analytics
        self.research = research
        self._research_snapshot = {"advisory_only": True, "status": "disabled"}
        self._research_last_publish = 0.0
        self._research_error = None
        self._inputs = queue.Queue(maxsize=queue_capacity)
        self._controls = queue.Queue(maxsize=128)
        self._invalidations = {}
        self._invalid_generations = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._shutdown_lock = threading.RLock()
        self._shutdown_started = False
        self._worker = None
        self._snapshot = {}
        self._status = {}
        self._last_publish = 0.0
        self._revision = 0
        self._command_cache = {}
        self._delays = deque(maxlen=10000)
        self._latency_bins = [0] * 60002
        self._latency_count = 0
        self._latency_max_ms = 0.0
        self._processed = 0
        self._overflow_count = 0
        self._incomplete = False
        self._last_error = None
        self.auto_start_pending = False
        self.provider_status = lambda: []
        self.on_overflow = lambda provider_id, generation: None
        if analytics is not None:
            engine.on("strategy_fill", analytics.on_fill)
            engine.on("funding", analytics.on_funding)

    def start(self):
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._shutdown_started = False
        self._publish_state(force=True)
        self._worker = threading.Thread(target=self._run, name="paper-account-writer", daemon=True)
        self._worker.start()

    def _enqueue(self, kind, payload, future=None, *, emergency=False):
        if self._shutdown_started:
            if future is not None:
                future.set_exception(RuntimeError("Paper application is shutting down"))
            return False
        target = self._controls if emergency else self._inputs
        try:
            target.put_nowait((time.monotonic(), kind, payload, future))
            return True
        except queue.Full:
            if future is not None:
                future.set_exception(RuntimeError("Paper input queue is full; retry after the backlog clears"))
            return False

    def enqueue_market(self, provider_id, event, generation=0):
        if self._stop.is_set() or self._shutdown_started:
            return False
        if self._enqueue("market", (provider_id, event, generation)):
            return True
        with self._lock:
            self._overflow_count += 1
            self._incomplete = True
            self._invalid_generations[provider_id] = max(generation, self._invalid_generations.get(provider_id, -1))
            self._invalidations[provider_id] = ("Market queue overflow: resynchronization required", generation)
        self.on_overflow(provider_id, generation)
        return False

    def invalidate_provider(self, provider_id, reason, generation=0):
        with self._lock:
            self._invalid_generations[provider_id] = max(generation, self._invalid_generations.get(provider_id, -1))
            prior = self._invalidations.get(provider_id)
            if prior is None or generation >= prior[1]:
                self._invalidations[provider_id] = (reason, generation)

    def bind_provider(self, provider_id, provider, instrument_ids):
        def bind():
            replace = getattr(self.engine, "replace_provider", None)
            if callable(replace):
                replace(provider_id, provider, instrument_ids)
            elif provider_id in self.engine._providers:
                # Engine compatibility: registration metadata is unchanged;
                # only the read-only connection instance changes on retry.
                self.engine._providers[provider_id] = provider
            else:
                self.engine.register_provider(provider_id, provider, instrument_ids)
        return self.call(bind)

    def call(self, callback, timeout=10.0):
        if threading.current_thread() is self._worker:
            return callback()
        future = Future()
        self._inputs.put((time.monotonic(), "call", callback, future), timeout=timeout)
        return future.result(timeout=timeout)

    def submit_command(self, envelope):
        future = Future()
        action = envelope.get("command", envelope.get("type")) if isinstance(envelope, dict) else None
        emergency = action in {"stop", "stop_strategy"}
        self._enqueue("command", envelope, future, emergency=emergency)
        return future

    def flush(self, timeout=10.0):
        """Barrier returning a fresh read model after all earlier normal input."""
        def publish():
            self._publish_state(force=True)
            return self.get_dashboard_state()
        return self.call(publish, timeout=timeout)

    def reset_latency_metrics(self):
        """Reset phase-wide bounded metrics on the authoritative writer."""
        def reset():
            self._delays.clear()
            self._latency_bins = [0] * 60002
            self._latency_count = 0
            self._latency_max_ms = 0.0
        if self._worker is None or threading.current_thread() is self._worker:
            return reset()
        return self.call(reset)

    def _record_queue_delay(self, milliseconds):
        delay = float(milliseconds)
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("queue delay must be finite and non-negative")
        self._delays.append(delay)
        # One-millisecond upper bounds plus a >60s overflow bin. This
        # aggregates the entire phase without growing with event count.
        self._latency_bins[min(60001, int(math.ceil(delay)))] += 1
        self._latency_count += 1
        self._latency_max_ms = max(self._latency_max_ms, delay)

    def _phase_delay_percentile(self):
        if not self._latency_count:
            return 0.0
        target = max(1, int(math.ceil(self._latency_count * 0.99)))
        total = 0
        for upper_bound, samples in enumerate(self._latency_bins):
            total += samples
            if total >= target:
                return self._latency_max_ms if upper_bound == 60001 else float(upper_bound)
        return self._latency_max_ms

    def tick(self, timestamp=None):
        # A full input queue must not quietly omit a timer. Collapse only
        # redundant timer wakeups; the next tick still processes missed work.
        return self._enqueue("timer", float(self.clock()) if timestamp is None else float(timestamp))

    def enqueue_provider_status(self, status):
        if not self._enqueue("provider", dict(status)):
            provider_id = status["provider_id"]
            self.invalidate_provider(provider_id, "Provider-state queue overflow", status.get("generation", 0))
            self._incomplete = True
            self.on_overflow(provider_id, status.get("generation", 0))

    def _record(self, kind, payload, **kwargs):
        if self.recorder is None:
            return
        try:
            self.recorder.append(kind, payload, processing_ts=self._writer_time, **kwargs)
        except Exception as exc:
            self._incomplete = True
            self._last_error = "Recording stopped: {}".format(type(exc).__name__)
            # Optional recording failure never discards financial journal
            # entries or fabricates a complete recording.
            try:
                self.recorder.close(complete=False)
            finally:
                self.recorder = None

    def _run(self):
        while not self._stop.is_set() or not self._inputs.empty() or not self._controls.empty():
            self._writer_time = max(self._writer_time, float(self.clock()))
            with self._lock:
                invalidations, self._invalidations = self._invalidations, {}
            for provider_id, (reason, generation) in invalidations.items():
                hook = getattr(self.engine, "invalidate_provider", None)
                if callable(hook):
                    hook(provider_id, reason, generation)
                self._research_call("invalidate_provider", provider_id)
                self._research_last_publish = 0.0
                self._record("provider", {"state": "invalidated", "reason": reason},
                             provider_id=provider_id, generation=generation)
            try:
                item = self._controls.get_nowait()
            except queue.Empty:
                try:
                    item = self._inputs.get(timeout=0.05)
                except queue.Empty:
                    self._publish_state()
                    continue
            arrived, kind, payload, future = item
            self._writer_time = max(self._writer_time, float(self.clock()))
            self._record_queue_delay((time.monotonic() - arrived) * 1000)
            try:
                if kind == "market":
                    provider_id, event, generation = payload
                    if generation > self._invalid_generations.get(provider_id, -1):
                        if self.recorder is not None:
                            try:
                                self.recorder.record_market(event, provider_id=provider_id, generation=generation,
                                                            processing_ts=self._writer_time)
                            except Exception:
                                self._incomplete = True
                                self.recorder.close(complete=False)
                                self.recorder = None
                        result = self.engine.on_market_event(event, provider_id=provider_id, generation=generation)
                        if result:
                            self._research_call("observe", event, self._writer_time,
                                                provider_id=provider_id, generation=generation)
                        if self.analytics is not None and result:
                            self.analytics.on_market(event)
                elif kind == "command":
                    result = self._execute_command(payload)
                elif kind == "timer":
                    self._record("timer", {"timestamp": self._writer_time})
                    self.engine.expire_stale_feeds(self._writer_time)
                    actions = getattr(self.engine, "process_account_actions", None)
                    if callable(actions):
                        actions(self._writer_time)
                    if (self.auto_start_pending and not self.engine.circuit_breaker_active
                            and self.engine.market_data_is_fresh()):
                        self.auto_start_pending = False
                        self._record("command", {"command": "start_strategy", "payload": {}, "automatic": True})
                        self.engine.start_strategy()
                    self._observe(self._writer_time)
                    result = None
                elif kind == "provider":
                    self._record("provider", payload, provider_id=payload["provider_id"],
                                 generation=payload.get("generation", 0))
                    self._research_call("provider_status", payload["provider_id"],
                                        payload.get("generation", 0), payload.get("state"))
                    result = None
                else:
                    result = payload()
                self._processed += 1
                self._publish_state(force=kind in {"command", "timer", "call"})
                if future is not None and not future.done():
                    future.set_result(result)
            except Exception as exc:
                self._last_error = "{}: {}".format(type(exc).__name__, str(exc)[:300])
                if future is not None and not future.done():
                    future.set_exception(exc)
            finally:
                # Queue task tracking is intentionally unused: Stop can
                # interrupt normal arrival processing through its own channel.
                pass

    def _research_call(self, method, *args, **kwargs):
        if self.research is None or self._research_error is not None:
            return None
        try:
            return getattr(self.research, method)(*args, **kwargs)
        except Exception as exc:
            # Advisory faults disable research only, never account processing.
            self._research_error = type(exc).__name__
            self._research_snapshot = {"advisory_only": True, "status": "error",
                                       "error": self._research_error}
            return None

    def _observe(self, timestamp):
        if self.analytics is None:
            return
        snapshot = self.engine.get_dashboard_state()
        portfolio = snapshot.get("portfolio")
        portfolio = dict(portfolio or {})
        if snapshot.get("portfolio_error"):
            portfolio.update(valuation_valid=False, valuation_error=snapshot["portfolio_error"])
        reporting = snapshot.get("reporting_currency", "USD")
        portfolio["fx_rates"] = {reporting: 1.0}
        for mark in snapshot.get("currency_conversion_marks", []):
            if not mark.get("stale"):
                portfolio["fx_rates"][mark["currency"]] = mark.get("rate", mark.get("rate_to_reporting"))
        revision = getattr(self.state_manager, "get_account_revision", None)
        if callable(revision):
            portfolio["account_revision"] = revision(self.account_id)
        self.analytics.observe(timestamp, portfolio, snapshot.get("instruments", []))

    def _publish_state(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_publish < 0.2:
            return
        try:
            snapshot = self.engine.get_dashboard_state()
            status = self.engine.get_status()
            if self.state_manager is not None:
                getter = getattr(self.state_manager, "get_account_revision", None)
                if callable(getter):
                    self._revision = getter(self.account_id)
            snapshot["application"] = {
                "protocol_version": self.PROTOCOL_VERSION, "account_id": self.account_id,
                "revision": self._revision, "live_acceptance": "pending",
                "queue_depth": self._inputs.qsize(), "queue_capacity": self._inputs.maxsize,
                "processed_inputs": self._processed, "overflow_count": self._overflow_count,
                "recording_complete": not self._incomplete, "last_error": self._last_error,
                "queue_delay_p99_ms": self._delay_percentile(),
                "queue_delay_window_sample_count": len(self._delays),
                "queue_delay_phase_p99_ms": self._phase_delay_percentile(),
                "queue_delay_phase_sample_count": self._latency_count,
                "queue_delay_phase_max_ms": self._latency_max_ms,
                "queue_delay_phase_histogram_resolution_ms": 1,
            }
            snapshot["provider_supervision"] = self.provider_status()
            if self.research is not None and now - self._research_last_publish >= 2.0:
                started = time.perf_counter()
                research = self._research_call("summary", self._writer_time)
                if research is not None:
                    research["status"] = "ready"
                    research["compute_ms"] = (time.perf_counter() - started) * 1000
                    if research["compute_ms"] > 50:
                        research["status"] = "budget_exceeded"
                        self._research_error = "ResearchComputeBudgetExceeded"
                        research["error"] = self._research_error
                    self._research_snapshot = research
                self._research_last_publish = now
            snapshot["research"] = self._research_snapshot
            if self.analytics is not None:
                snapshot["analytics"] = self.analytics.summary()
            with self._lock:
                self._snapshot, self._status = snapshot, status
            self._last_publish = now
        except Exception as exc:
            self._last_error = "Account read model unavailable: {}".format(type(exc).__name__)

    def _delay_percentile(self):
        values = sorted(self._delays)
        return values[min(len(values) - 1, int(len(values) * 0.99))] if values else 0.0

    def _execute_command(self, envelope):
        """Record the exact command outcome, including rejects and retries.

        Outcomes are recorded after dispatch but before another writer input;
        replay can reproduce partially applied imports and distinguish a
        rejected revision or cached request from an executed command.
        """
        try:
            result, duplicate = self._execute_command_inner(envelope)
        except Exception as exc:
            self._record("command", {"envelope": envelope, "outcome": {
                "success": False, "code": "validation_error",
                "error_type": type(exc).__name__, "message": str(exc)[:500]}})
            raise
        self._record("command", {"envelope": envelope, "outcome": result,
                                 "duplicate": duplicate})
        return result

    def _execute_command_inner(self, envelope):
        if not isinstance(envelope, dict) or envelope.get("protocol_version") != 1:
            raise ValueError("Unsupported paper command protocol")
        request_id = envelope.get("request_id")
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError("request_id must contain 1–128 characters")
        if envelope.get("account_id") != self.account_id:
            raise ValueError("Paper command account does not match this application")
        command = envelope.get("command")
        payload = envelope.get("payload", {})
        if not isinstance(command, str) or not isinstance(payload, dict):
            raise ValueError("command and payload must be a name and object")
        encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False)
        request_hash = hashlib.sha256(encoded.encode()).hexdigest()
        manager = self.state_manager
        get_result = getattr(manager, "get_command_result", None)
        cached = self._command_cache.get(request_id)
        if cached is None and callable(get_result):
            cached = get_result(self.account_id, request_id)
        if cached is not None:
            if cached.get("request_hash", request_hash) != request_hash:
                raise ValueError("request_id was reused for a different command")
            return copy.deepcopy(cached), True
        scope = getattr(manager, "command_scope", None)
        with scope(self.account_id, request_id, request_hash) if callable(scope) else nullcontext():
            expected = envelope.get("expected_revision")
            if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
                raise ValueError("expected_revision must be a non-negative integer")
            getter = getattr(manager, "get_account_revision", None)
            actual = getter(self.account_id) if callable(getter) else self._revision
            self._revision = actual
            if command not in self.READ_COMMANDS | {"stop", "stop_strategy"} and expected != actual:
                result = {"success": False, "code": "revision_conflict", "message":
                          "Account state changed. Review the latest state and send a new request."}
            else:
                try:
                    dispatch_payload = dict(payload)
                    if command == "submit_paper_order":
                        dispatch_payload["request_id"] = request_id
                    value = self._dispatch(command, dispatch_payload)
                    result = {"success": True, "code": "ok", "data": value}
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    result = {"success": False, "code": "rejected", "message": str(exc)[:500]}
            result.update({"protocol_version": 1, "request_id": request_id, "account_id": self.account_id,
                           "request_hash": request_hash, "action": command})
            self._command_receipts(result, actual, getter)
            # Preserve the semantic dispatch result before fallible final work.
            self._stage_command_outcome(result)
            if result["success"] and command not in self.READ_COMMANDS:
                try:
                    checkpoint = getattr(self.engine, "_save_checkpoint", None)
                    if callable(checkpoint):
                        checkpoint(self._writer_time)
                    if not callable(getter):
                        self._revision = actual + 1
                except Exception as exc:
                    if not result.get("committed_effects"):
                        self._incomplete = True
                    self._command_persistence_error(result, "checkpoint", exc)
            self._command_receipts(result, actual, getter)
            if getattr(self.engine, "circuit_breaker_reason", None) == "paper_history_persistence_failed":
                result["persistence_halt"] = self.engine.circuit_breaker_reason
            self._stage_command_outcome(result)
            outcome_saver = getattr(manager, "save_command_outcome", None)
            saver = getattr(manager, "save_command_result", None)
            if callable(saver):
                try:
                    saver(self.account_id, request_id, request_hash, result)
                except Exception as exc:
                    self._command_persistence_error(result, "ack", exc)
                    if callable(outcome_saver):
                        try:
                            outcome_saver(self.account_id, request_id, request_hash, result)
                        except Exception:
                            self._incomplete = True
                    else:
                        self._incomplete = True
            while len(self._command_cache) > 4096:
                self._command_cache.pop(next(iter(self._command_cache)))
            self._command_cache[request_id] = copy.deepcopy(result)
            return result, False

    def _command_receipts(self, result: dict[str, Any], prior_revision: int,
                          revision_getter: Callable[[str], int] | None) -> None:
        receipts = getattr(self.state_manager, "current_command_effects", None)
        effects = receipts() if callable(receipts) else None
        if effects is not None:
            self._revision = effects[-1]["account_seq"] if effects else prior_revision
            result["revision"] = self._revision
            result["committed_effects"] = [{key: value for key, value in effect.items() if key != "account_seq"}
                                           for effect in effects if effect["event_type"] != "account_checkpoint"]
        elif callable(revision_getter):
            result.update(revision=None, revision_status="unverified")
            self._incomplete = True
        else:
            result["revision"] = self._revision

    def _stage_command_outcome(self, result: dict[str, Any]) -> None:
        self._command_cache[result["request_id"]] = copy.deepcopy(result)
        saver = getattr(self.state_manager, "save_command_outcome", None)
        if callable(saver):
            try:
                saver(self.account_id, result["request_id"], result["request_hash"], result)
            except Exception as exc:
                self._incomplete = True
                self._command_persistence_error(result, "outcome", exc)

    def _command_persistence_error(self, result: dict[str, Any], stage: str, error: Exception) -> None:
        # Preserve the exact dispatch decision, including partial rejection.
        result.setdefault("persistence_error", {"stage": stage, "error_type": type(error).__name__,
                                                 "message": str(error)[:500]})
        self._last_error = f"Command {stage} persistence failed: {type(error).__name__}"

    def _dispatch(self, command, data):
        engine = self.engine
        command = {"update_parameters": "update_params", "set_parameters": "update_params",
                   "acknowledge_risk_halt": "acknowledge_risk_circuit_breaker"}.get(command, command)
        sourced_types = {"apply_equity_split": "split", "apply_equity_dividend": "dividend",
                         "record_fx_financing": "financing", "settle_future": "settlement"}
        if command in sourced_types:
            source = data.get("source")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("A source reference is required for every accounting adjustment")
            action_id = data.get("action_id", data.get("adjustment_id"))
            values = {key: value for key, value in data.items() if key not in {
                "source", "action_id", "adjustment_id", "instrument_id", "timestamp"}}
            return engine.schedule_account_action({"action_id": action_id,
                "action_type": sourced_types[command], "instrument_id": data.get("instrument_id"),
                "source": source, "effective_ts": data.get("timestamp", self._writer_time), "payload": values})
        if command in {"start", "start_strategy", "stop", "stop_strategy"}:
            self.auto_start_pending = False
            getattr(engine, "start_strategy" if command.startswith("start") else "stop_strategy")()
            return {"message": "Paper strategy {}".format("started" if command.startswith("start") else "stopped")}
        if command in {"reset", "reset_strategy"}:
            raise ValueError("Account reset is disabled; use an explicit new paper account")
        if command == "acknowledge_risk_circuit_breaker":
            self.auto_start_pending = False
            recovery_halt = getattr(self.state_manager, "get_command_persistence_halt", None)
            if callable(recovery_halt) and recovery_halt(self.account_id) == "command_outcome_reconciliation_required":
                raise ValueError("committed command effects require outcome reconciliation before risk can resume")
            return engine.acknowledge_risk_circuit_breaker()
        if command == "update_params":
            engine.update_parameters(data.get("params", data))
            return {"updated": data.get("params", data)}
        if command == "roll_future":
            kwargs = {"quantity": data["quantity"]} if "quantity" in data else {}
            return engine.roll_future(data.get("old_instrument_id"), data.get("new_instrument_id"), **kwargs)
        if command == "submit_paper_order":
            return engine.submit_paper_order(data.get("instrument_id"), data.get("side"),
                data.get("quantity"), data.get("price"), request_id=data.get("request_id"))
        if command == "record_funding":
            return engine.record_funding(**data)
        if command == "settle_future":
            return engine.settle_future(data.get("instrument_id"), data.get("settlement_price"), data.get("timestamp"))
        if command == "apply_equity_split":
            return engine.apply_equity_split(data.get("action_id"), data.get("instrument_id"),
                                             data.get("ratio"), timestamp=data.get("timestamp"))
        if command == "apply_equity_dividend":
            return engine.apply_equity_dividend(data.get("action_id"), data.get("instrument_id"),
                data.get("amount_per_share"), data.get("eligible_quantity"), timestamp=data.get("timestamp"))
        if command == "record_fx_financing":
            return engine.record_fx_financing(data.get("adjustment_id"), data.get("instrument_id"),
                data.get("amount"), data.get("currency"), timestamp=data.get("timestamp"), kind=data.get("kind", "fx_financing"))
        if command == "schedule_account_action":
            return engine.schedule_account_action(data)
        if command == "import_account_actions":
            return engine.import_account_actions(data.get("actions", []))
        if command == "get_account_actions":
            return engine.get_account_actions()
        if command in {"get_status", "status_request"}:
            return self.get_dashboard_state()
        if command == "get_trades":
            limit = max(0, min(2000, int(data.get("limit", 50))))
            return {"trades": engine.trades[-limit:] if limit else [], "total": len(engine.trades)}
        if command == "get_history":
            if self.analytics is not None:
                return self.analytics.history(start=data.get("start"), end=data.get("end"), limit=min(10000, int(data.get("limit", 1000))))
            return self.state_manager.get_account_history(self.account_id, data.get("start"), data.get("end"), min(10000, int(data.get("limit", 1000))))
        if command == "preview_migration":
            return self.state_manager.preview_account_migration(self.account_id, data.get("config", {}))
        if command == "get_diagnostics":
            return {"account_id": self.account_id, "revision": self._revision,
                    "execution_mode": "local_paper", "live_acceptance": "pending",
                    "application": self.get_dashboard_state().get("application", {}),
                    "providers": self.provider_status()}
        raise ValueError("Unknown paper command: {}".format(command))

    def get_dashboard_state(self):
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def get_status(self):
        with self._lock:
            return copy.deepcopy(self._status)

    get_full_status = get_status

    @property
    def circuit_breaker_active(self):
        return bool(self.get_dashboard_state().get("circuit_breaker_active"))

    def market_data_is_fresh(self):
        return any(row.get("market_data_ready") for row in self.get_dashboard_state().get("instruments", []))

    def stop(self):
        with self._shutdown_lock:
            worker = self._worker
            if worker is None:
                return
            if threading.current_thread() is worker:
                raise RuntimeError("Paper application shutdown must be requested outside its writer")
            self._shutdown_started = True
            try:
                if worker.is_alive():
                    # A long financial/backup operation must finish before
                    # the enclosing account lease can close its connections.
                    self.call(self.engine.stop_strategy, timeout=None)
            finally:
                self._stop.set()
                worker.join()
                self._worker = None
                if self.recorder is not None:
                    self.recorder.close(complete=not self._incomplete)
