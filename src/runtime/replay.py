"""Deterministic virtual-clock execution of recorded local paper sessions."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from ..config.cross_asset_config import load_cross_asset_config, parse_cross_asset_config
from ..connectors.market_data import ProviderCapabilities, ProviderState, ProviderStatus
from .recording import (
    EXECUTION_SEMANTICS_VERSION,
    QUOTE_POLICY_VERSION,
    canonical_json,
    event_from_dict,
    iter_records,
    load_manifest,
    sanitize,
)


class VirtualClock:
    def __init__(self, timestamp: float = 0.0):
        self.value = float(timestamp)

    def __call__(self) -> float:
        return self.value

    def advance_to(self, timestamp: float) -> None:
        value = float(timestamp)
        if not math.isfinite(value) or value < self.value:
            raise ValueError("replay clock cannot move backwards")
        self.value = value


class DeterministicIds:
    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.sequence = 0

    def __call__(self, prefix: str = "event") -> str:
        self.sequence += 1
        digest = hashlib.sha256("{}:{}:{}".format(self.seed, prefix, self.sequence).encode()).hexdigest()
        return "{}:{}".format(prefix, digest[:24])


@dataclass(frozen=True)
class FaultSpec:
    """Explicit, deterministic mutations; result records the fault scenario."""
    drop_every: int = 0
    duplicate_every: int = 0
    source_clock_offset: float = 0.0
    delivery_delay: float = 0.0

    def __post_init__(self):
        if self.drop_every < 0 or self.duplicate_every < 0:
            raise ValueError("fault intervals cannot be negative")
        if (not math.isfinite(self.source_clock_offset) or not math.isfinite(self.delivery_delay)
                or self.delivery_delay < 0):
            raise ValueError("fault delays must be finite; delivery_delay cannot be negative")


class _ReplayProvider:
    def __init__(self, name: str):
        self.name = name
        self.capabilities = ProviderCapabilities(quote=True, order_book=True, trades=True, sessions=True)
        self.status = ProviderStatus(provider=name, state=ProviderState.CONNECTED)

    def start(self, on_event):
        self.status = ProviderStatus(provider=self.name, state=ProviderState.CONNECTED)

    def stop(self):
        self.status = ProviderStatus(provider=self.name, state=ProviderState.STOPPED)


class _InitialStateManager:
    """Replay-only checkpoint adapter; never opens the original account DB."""
    def __init__(self, checkpoint: dict[str, Any] | None, timestamp: float,
                 signature_history: Path | None = None) -> None:
        self.checkpoint = copy.deepcopy(checkpoint)
        self.timestamp = timestamp
        self.signature_history = signature_history
        self.effects: list[dict[str, Any]] = []

    def load_cross_asset_snapshot(self, account_id):
        if self.checkpoint is None:
            return None
        return {"timestamp": self.timestamp, "snapshot": copy.deepcopy(self.checkpoint)}

    def commit_cross_asset_event(self, account_id, event_id, event_type, timestamp, payload,
                                 snapshot, snapshot_timestamp=None):
        if event_type != "account_checkpoint":
            self.effects.append({"event_id": event_id, "event_type": event_type,
                                 "effective_timestamp": float(timestamp), "payload": sanitize(payload)})
        return True

    def save_cross_asset_snapshot(self, *args, **kwargs):
        return None

    def get_cross_asset_signature(self, account_id, namespace, key):
        value = (self.checkpoint or {}).get("ledger", {}).get(namespace, {}).get(key)
        if value is None and self.signature_history is not None:
            with closing(sqlite3.connect(self.signature_history.as_uri() + "?mode=ro", uri=True)) as conn:
                row = conn.execute("SELECT signature_json FROM signatures WHERE namespace=? AND event_id=?",
                                   (namespace, key)).fetchone()
            if row is not None:
                value = json.loads(row[0])
                if not isinstance(value, list):
                    raise ValueError("recording signature history requires signature arrays")
        return None if value is None else tuple(value)


class ReplayRunner:
    def __init__(self, config: Any, *, engine_factory: Optional[Callable] = None,
                 strategy_factory: Optional[Callable] = None, seed: Optional[int] = None,
                 research_assumption_override: bool = False):
        self.config = load_cross_asset_config(config) if isinstance(config, (str, Path)) else config
        if engine_factory is None:
            from .cross_asset_paper_engine import CrossAssetPaperEngine
            engine_factory = CrossAssetPaperEngine
        self.engine_factory = engine_factory
        self.strategy_factory = strategy_factory
        self.seed = None if seed is None else int(seed)
        self.research_assumption_override = research_assumption_override

    def run(self, directory: Union[str, Path], *, speed: float = 0.0,
            faults: Optional[FaultSpec] = None, allow_incomplete: bool = False,
            record_filter: Optional[Callable[[Dict], bool]] = None,
            until_ts: Optional[float] = None) -> Dict[str, Any]:
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("speed must be non-negative (zero means maximum speed)")
        manifest = load_manifest(directory, allow_incomplete=allow_incomplete)
        recorded_semantics = manifest.get("execution_semantics_version", 1)
        recorded_quote_policy = manifest.get("quote_policy_version", 1)
        quote_policy_overridden = recorded_quote_policy != QUOTE_POLICY_VERSION
        initial_semantics = None
        if manifest.get("initial_state") is not None:
            initial_semantics = (manifest["initial_state"].get("engine", {}).get("execution_assumptions") or {}).get(
                "_portfolio", {}).get("execution_semantics_version", 1)
            if type(initial_semantics) is not int or initial_semantics < 1:
                raise ValueError("initial checkpoint execution semantics version must be a positive integer")
        semantics_overridden = (recorded_semantics != EXECUTION_SEMANTICS_VERSION
                                or initial_semantics not in (None, EXECUTION_SEMANTICS_VERSION))
        recorded_configuration = parse_cross_asset_config(manifest["config"])
        execution_config_hash = hashlib.sha256(canonical_json(sanitize(asdict(self.config))).encode()).hexdigest()
        recorded_execution_hash = hashlib.sha256(canonical_json(sanitize(asdict(recorded_configuration))).encode()).hexdigest()
        if execution_config_hash != recorded_execution_hash and not self.research_assumption_override:
            raise ValueError("replay configuration differs from recording; research overrides must be explicit")
        if semantics_overridden and not self.research_assumption_override:
            raise ValueError("recording execution semantics differ; corrected re-evaluation requires research_assumption_override=True")
        if quote_policy_overridden and not self.research_assumption_override:
            raise ValueError("recording quote policy differs; corrected re-evaluation requires research_assumption_override=True")
        records = iter_records(directory, allow_incomplete=allow_incomplete)
        first = next(records, None)
        if first is None:
            raise ValueError("recording is empty")
        clock = VirtualClock(first["processing_ts"])
        seed = int(manifest["seed"]) if self.seed is None else self.seed
        ids = DeterministicIds(seed)
        parameters = inspect.signature(self.engine_factory).parameters
        if "id_factory" not in parameters:
            raise TypeError("engine_factory must support deterministic id_factory")
        kwargs = {"clock": clock, "id_factory": ids}
        history = manifest.get("initial_signature_history")
        signature_history = (Path(directory).resolve() / history["file"]) if history is not None else None
        replay_store = _InitialStateManager(manifest.get("initial_state"), clock.value, signature_history)
        if manifest.get("initial_state") is not None:
            kwargs["state_manager"] = replay_store
        if self.research_assumption_override:
            kwargs["allow_config_migration"] = True
        if self.strategy_factory is not None:
            kwargs["strategy_factory"] = self.strategy_factory
        engine = self.engine_factory(self.config, **kwargs)
        engine.state_manager = replay_store
        from .portfolio_analytics import PortfolioAnalytics
        analytics = PortfolioAnalytics(account_id=self.config.account_id,
                                       max_mark_age=self.config.market_data_stale_after)
        engine.on("strategy_fill", analytics.on_fill)
        engine.on("funding", analytics.on_funding)
        from .application_service import PaperApplicationService
        command_router = PaperApplicationService(engine, clock=clock)
        providers = {}
        bindings = manifest.get("provider_bindings") or {
            provider_id: list(item.instrument_ids) for provider_id, item in self.config.providers.items()}
        for provider_id, instrument_ids in bindings.items():
            provider = _ReplayProvider(provider_id)
            providers[provider_id] = provider
            engine.register_provider(provider_id, provider, instrument_ids)
        trace = []
        observations = []
        command_outcomes = []
        last_observation = None
        for event_name in ("order_accepted", "order_rejected", "strategy_fill", "orders_cancelled",
                           "corporate_action", "fx_financing", "futures_settlement", "futures_roll"):
            engine.on(event_name, lambda payload, name=event_name: trace.append({"type": name, "payload": payload}))
        count = market_count = dropped = duplicates = 0
        faults = faults or FaultSpec()
        previous = clock.value
        from itertools import chain
        for record in chain((first,), records):
            if until_ts is not None and float(record["processing_ts"]) > until_ts:
                break
            stamp = max(clock.value, float(record["processing_ts"]) + faults.delivery_delay)
            if speed:
                time.sleep(max(0.0, stamp - previous) / speed)
            clock.advance_to(stamp)
            command_router._writer_time = clock.value
            previous = stamp
            count += 1
            if record_filter is not None and not record_filter(record):
                continue
            kind = record["kind"]
            payload = record["payload"]
            if kind == "market":
                market_count += 1
                if faults.drop_every and market_count % faults.drop_every == 0:
                    dropped += 1
                    continue
                event = event_from_dict(payload)
                event = replace(event, event_ts=max(0.0, event.event_ts + faults.source_clock_offset),
                                receive_ts=event.receive_ts + faults.delivery_delay)
                provider_id = record.get("provider_id")
                generation = int(record.get("generation", 0))
                if provider_id:
                    accepted = engine.on_market_event(event, provider_id=provider_id, generation=generation)
                else:
                    accepted = engine.on_market_event(event)
                if accepted:
                    analytics.on_market(event)
                if faults.duplicate_every and market_count % faults.duplicate_every == 0:
                    engine.on_market_event(event, **({"provider_id": provider_id, "generation": generation}
                                                    if provider_id else {}))
                    duplicates += 1
            elif kind == "timer":
                engine.expire_stale_feeds(clock.value)
                if callable(getattr(engine, "process_account_actions", None)):
                    engine.process_account_actions(clock.value)
            elif kind == "command":
                self._command(engine, payload, command_router)
                if isinstance(payload.get("outcome"), dict):
                    command_outcomes.append(copy.deepcopy(payload["outcome"]))
            elif kind == "provider":
                provider_id = record.get("provider_id") or payload.get("provider_id")
                state = str(payload.get("state", "unknown"))
                if provider_id in providers:
                    good = state in {"connected", "healthy"}
                    providers[provider_id].status = ProviderStatus(
                        provider=provider_id, state=ProviderState.CONNECTED if good else ProviderState.ERROR)
                    if state == "invalidated":
                        engine.invalidate_provider(provider_id, payload.get("reason", state),
                                                   int(record.get("generation", 0)))
            elif kind == "configuration":
                if payload.get("parameters"):
                    engine.update_parameters(payload["parameters"])
                elif payload.get("action") != "manifest":
                    raise ValueError("configuration transition requires replayable parameters")
            if last_observation is None or clock.value - last_observation >= 1.0:
                try:
                    state = engine.get_dashboard_state()
                    portfolio = dict(state.get("portfolio") or {})
                    portfolio["fx_rates"] = {self.config.reporting_currency: 1.0}
                    for mark in state.get("currency_conversion_marks", ()):
                        if not mark.get("stale"):
                            portfolio["fx_rates"][mark["currency"]] = mark.get("rate", mark.get("rate_to_reporting"))
                    analytics.observe(clock.value, portfolio, state.get("instruments", ()))
                    observations.append(analytics.history(limit=1)[0])
                except (ValueError, KeyError):
                    observations.append({"timestamp": clock.value, "equity": None,
                                         "valuation_valid": False})
                last_observation = clock.value
        # Explicitly release local working quotes; all price inputs remain unchanged.
        engine.stop_strategy()
        try:
            snapshot = engine.ledger.snapshot(as_of=clock.value)
            financial = {"cash": dict(snapshot.cash_by_currency),
                     "positions": {key: {"quantity": value.quantity,
                         "average_entry_price": value.average_entry_price,
                         "realized_pnl": value.realized_pnl} for key, value in snapshot.positions.items()},
                     "equity": snapshot.equity, "total_pnl": snapshot.total_pnl,
                     "fees_paid": snapshot.fees_paid, "costs_paid": snapshot.costs_paid,
                         "income": snapshot.income, "valuation_valid": True}
        except (ValueError, KeyError) as exc:
            saved = engine.ledger.export_state()
            financial = {"cash": saved["cash"], "positions": saved["positions"],
                         "equity": None, "total_pnl": None, "valuation_valid": False,
                         "valuation_error": str(exc)}
        evidence = {"trace": trace, "financial": financial, "committed_effects": replay_store.effects}
        return {"schema_version": 1, "records_processed": count, "market_records": market_count,
                "dropped_records": dropped, "duplicate_records": duplicates,
                "input_complete": bool(manifest["complete"]), "seed": seed,
                "faults": faults.__dict__, "trace": trace, "financial": financial,
                "digest": hashlib.sha256(canonical_json(evidence).encode()).hexdigest(),
                "fills": list(engine.trades), "observations": observations,
                "command_outcomes": command_outcomes,
                "committed_effects": replay_store.effects,
                "analytics": analytics.summary(),
                "config_hash": execution_config_hash,
                "execution_config_hash": execution_config_hash,
                "recording_config_hash": manifest["config_hash"],
                "recorded_execution_config_hash": recorded_execution_hash,
                "assumptions_overridden": (execution_config_hash != recorded_execution_hash
                                           or semantics_overridden or quote_policy_overridden),
                "execution_semantics_version": EXECUTION_SEMANTICS_VERSION,
                "recorded_execution_semantics_version": recorded_semantics,
                "recorded_initial_execution_semantics_version": initial_semantics,
                "execution_semantics_overridden": semantics_overridden,
                "quote_policy_version": QUOTE_POLICY_VERSION,
                "recorded_quote_policy_version": recorded_quote_policy,
                "quote_policy_overridden": quote_policy_overridden,
                "replay_mode": ("corrected_re_evaluation" if (semantics_overridden
                                or quote_policy_overridden or execution_config_hash != recorded_execution_hash)
                                else "current_semantics"),
                "validation_status": "offline_replay_only"}

    @staticmethod
    def _command(engine: Any, payload: Dict[str, Any], router=None) -> None:
        """Use the application's dispatcher and preserve recorded decisions.

        A revision-conflicted command or an acknowledged retry did not execute
        in the captured session. A dispatched-but-rejected import may have
        committed earlier valid actions, so execute it and verify rejection.
        Older fixture records without an outcome remain supported.
        """
        from .application_service import PaperApplicationService
        if router is None:
            router = PaperApplicationService(engine, clock=engine.clock)
        wrapped = isinstance(payload.get("envelope"), dict)
        envelope = payload["envelope"] if wrapped else payload
        outcome = payload.get("outcome") if wrapped else {}
        if not isinstance(outcome, dict):
            raise ValueError("wrapped replay command requires its recorded outcome")
        if wrapped and (payload.get("duplicate") or outcome.get("code") in {
                "revision_conflict", "validation_error", "interrupted_before_commit"}):
            return
        action = envelope.get("command", envelope.get("command_type", envelope.get("type", envelope.get("action"))))
        data = dict(envelope.get("payload", envelope.get("parameters", {})))
        # Older restored-account recordings may lack durable signature history.
        if wrapped and action == "record_funding" and isinstance(outcome, dict) and outcome.get("success") and outcome.get("data") is False:
            return
        if action in PaperApplicationService.READ_COMMANDS | {"diagnostics", "migration_preview", "ping", "authenticate"}:
            return
        if action == "submit_paper_order":
            data["request_id"] = envelope.get("request_id", data.get("request_id"))
        effects = getattr(engine.state_manager, "effects", None)
        before_effects = len(effects) if effects is not None else 0
        rejected = False
        try:
            value = router._dispatch(action, data)
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            if wrapped and outcome.get("success") is False and outcome.get("code") == "rejected":
                rejected = True
            else:
                raise ValueError(f"replay command diverged: {action}: {exc}") from exc
        if wrapped and outcome.get("success") is False and not rejected:
            raise ValueError(f"replay command unexpectedly succeeded: {action}")
        if wrapped and not rejected and action == "record_funding" and value != outcome.get("data"):
            raise ValueError("replay funding application differs from its recorded outcome")
        if wrapped and "committed_effects" in outcome:
            observed_effects = effects[before_effects:] if effects is not None else []
            if canonical_json(observed_effects) != canonical_json(outcome["committed_effects"]):
                raise ValueError(f"replay committed effects differ from recorded command: {action}")
        if wrapped and outcome.get("persistence_halt"):
            engine.circuit_breaker_active = True
            engine.circuit_breaker_reason = outcome["persistence_halt"]
            engine.is_running = False
            engine._cancel_all_quotes(reason=outcome["persistence_halt"])
