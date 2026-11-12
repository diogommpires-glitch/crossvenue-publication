"""Chronological, reproducible comparisons of paper quoting policies.

Candidate selection sees training outcomes only. Validation is reported before
the frozen candidate is run on the untouched test interval. No policy is
automatically promoted into a running account.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import statistics
from typing import Any, Dict, Iterable, Optional

from ..strategies.adaptive_market_maker import adaptive_strategy_factory
from .recording import canonical_json, iter_records, load_manifest
from .replay import ReplayRunner


def _metrics(result: Dict[str, Any], start: float, end: float) -> Dict[str, Any]:
    observations = [row for row in result["observations"] if start <= row["timestamp"] <= end]
    rows = [row for row in observations if row.get("valuation_valid")]
    fills = [fill for fill in result["fills"] if start <= float(fill["timestamp"]) <= end]
    if len(rows) < 2:
        return {"valid": False, "reason": "insufficient_valid_valuation_observations", "fills": len(fills)}
    first, last = rows[0], rows[-1]
    interval = [row for row in observations if first["timestamp"] < row["timestamp"] <= last["timestamp"]]
    gap = any(not row.get("valuation_valid") for row in interval)
    net = last["equity"] - first["equity"] - sum(row.get("external_flow", 0) for row in interval)
    if gap or any(row["equity"] <= 0 for row in rows):
        # Endpoints and all cash flows identify net PnL, but an unknown
        # intermediate valuation cannot identify time-weighted risk/return.
        return {"valid": False, "reason": "valuation_gap" if gap else "nonpositive_equity",
                "start": first["timestamp"], "end": last["timestamp"], "net_pnl": net,
                "max_drawdown": None, "return": None, "score": None, "fills": len(fills),
                "observations": len(rows), "all_execution_estimated": True}
    peak = 1.0
    wealth = 1.0
    drawdown = 0.0
    prior = first
    inventory_time = {}
    for row in rows:
        if row is not first:
            wealth *= (row["equity"] - row.get("external_flow", 0)) / prior["equity"]
            peak = max(peak, wealth)
            drawdown = max(drawdown, 1 - wealth / peak)
            for identity, quantity in prior.get("positions", {}).items():
                if abs(quantity) > 1e-12:
                    inventory_time[identity] = inventory_time.get(identity, 0) + row["timestamp"] - prior["timestamp"]
        prior = row
    # Quantity exposures are retained by instrument; they are never summed
    # into a fake cross-asset dollar risk measure.
    max_inventory = {}
    for row in rows:
        for instrument, quantity in row.get("positions", {}).items():
            max_inventory[instrument] = max(max_inventory.get(instrument, 0), abs(quantity))
    fee_delta = None if first.get("fees_paid") is None or last.get("fees_paid") is None else last["fees_paid"] - first["fees_paid"]
    cost_delta = None if first.get("costs_paid") is None or last.get("costs_paid") is None else last["costs_paid"] - first["costs_paid"]
    turnover = {}
    for fill in fills:
        instrument = fill["instrument_id"]
        turnover[instrument] = turnover.get(instrument, 0) + abs(float(fill["quantity"]))
    markouts = {}
    for horizon in (1, 5, 30):
        values = [entry["signed_markout_bps"] for row in rows for entry in row.get("markout_observations", ())
                  if entry["horizon_seconds"] == horizon and entry["signed_markout_bps"] is not None]
        markouts[str(horizon)] = {"count": len(values), "mean_bps": statistics.mean(values) if values else None}
    return {"valid": True, "start": first["timestamp"], "end": last["timestamp"],
            "net_pnl": net, "max_drawdown": drawdown, "return": wealth - 1, "fees_paid": fee_delta,
            "carry_costs": cost_delta, "fills": len(fills),
            "max_inventory_by_instrument": max_inventory, "turnover_by_instrument": turnover,
            "inventory_time_seconds": inventory_time, "markouts": markouts,
            "max_gross_exposure": max((row["gross_exposure"] for row in rows if row.get("gross_exposure") is not None), default=None),
            "score": net - first["equity"] * drawdown,
            "observations": len(rows), "all_execution_estimated": True}


def evaluate_recording(directory, config, *, seed: int = 0,
                       candidates: Optional[Iterable[Dict[str, float]]] = None,
                       max_drawdown: float = 0.05) -> Dict[str, Any]:
    if not 0 < max_drawdown < 1:
        raise ValueError("max_drawdown must be a fraction between zero and one")
    manifest = load_manifest(directory)
    bounds = [record["processing_ts"] for record in iter_records(directory)]
    if len(bounds) < 5 or bounds[-1] <= bounds[0]:
        raise ValueError("research needs a non-empty chronological recording")
    first, last = float(bounds[0]), float(bounds[-1])
    duration = last - first
    train_end, validation_end = first + duration * 0.6, first + duration * 0.8
    grid = list(candidates) if candidates is not None else [
        {"volatility_multiplier": volatility, "adverse_selection_multiplier": adverse}
        for volatility in (1.0, 2.0) for adverse in (0.0, 1.0)]
    if not grid:
        raise ValueError("at least one training candidate is required")
    trials = []
    for parameters in sorted(grid, key=canonical_json):
        trial = ReplayRunner(config, strategy_factory=adaptive_strategy_factory(parameters), seed=seed,
                             research_assumption_override=True).run(
            directory, until_ts=train_end)
        metrics = _metrics(trial, first, train_end)
        feasible = bool(metrics.get("valid") and metrics["max_drawdown"] <= max_drawdown)
        trials.append({"parameters": parameters, "training": metrics, "feasible": feasible,
                       "training_digest": trial["digest"]})
    eligible = [trial for trial in trials if trial["feasible"]]
    if not eligible:
        return {"schema_version": 1, "status": "insufficient_or_infeasible_training",
                "trials": trials, "promotion_eligible": False,
                "split": {"training": [first, train_end], "validation": [train_end, validation_end],
                          "test": [validation_end, last]}}
    selected = max(eligible, key=lambda trial: trial["training"]["score"])
    validation = ReplayRunner(config, strategy_factory=adaptive_strategy_factory(selected["parameters"]),
                              seed=seed, research_assumption_override=True).run(directory, until_ts=validation_end)
    validation_metrics = _metrics(validation, train_end, validation_end)
    # Freeze the candidate before opening the test interval. Test outcomes
    # are never used to select a different coefficient combination.
    frozen_policy = {"policy_version": "adaptive_v1", "parameters": selected["parameters"],
                     "training_config_hash": manifest["config_hash"], "seed": seed}
    policy_hash = hashlib.sha256(canonical_json(frozen_policy).encode()).hexdigest()
    scenarios = {}
    from ..config.cross_asset_config import load_cross_asset_config
    from pathlib import Path
    parsed = load_cross_asset_config(config) if isinstance(config, (str, Path)) else config
    for name, scenario_config in (
        ("configured", parsed),
        ("conservative_queue_and_latency", replace(parsed, instruments={key: replace(
            item, queue_fraction=min(item.queue_fraction, 0.05), maker_latency_ms=item.maker_latency_ms * 2)
            for key, item in parsed.instruments.items()})),
    ):
        baseline = ReplayRunner(scenario_config, seed=seed, research_assumption_override=True).run(directory)
        candidate = ReplayRunner(scenario_config, strategy_factory=adaptive_strategy_factory(
            selected["parameters"]), seed=seed, research_assumption_override=True).run(directory)
        scenarios[name] = {"baseline_test": _metrics(baseline, validation_end, last),
                           "candidate_test": _metrics(candidate, validation_end, last),
                           "baseline_digest": baseline["digest"], "candidate_digest": candidate["digest"]}
    return {"schema_version": 1, "status": "offline_evaluation_complete",
            "split": {"training": [first, train_end], "validation": [train_end, validation_end],
                      "test": [validation_end, last]}, "trials": trials,
            "selected_policy": frozen_policy, "policy_hash": policy_hash,
            "validation": validation_metrics, "scenarios": scenarios,
            "calibration_status": "unvalidated_until_representative_recorded_data_review",
            "promotion_eligible": False,
            "promotion_reason": "explicit_operator_review_and_live_data_acceptance_required",
            "profitability_guaranteed": False}
