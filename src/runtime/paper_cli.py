"""Command line entry point for local paper recording, replay, and research."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ..config.cross_asset_config import load_cross_asset_config
from .recording import canonical_json, sanitize


def _default_config() -> Path:
    return Path(__file__).parents[2] / "config" / "cross_asset_paper.example.json"


def diagnostics(config_path=None):
    """Inspect installation/configuration without connecting or revealing keys."""
    config = load_cross_asset_config(config_path or _default_config())
    versions = {}
    for package in ("numpy", "websockets", "requests", "databento"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    providers = []
    for identity, provider in config.providers.items():
        variables = (["DATABENTO_API_KEY"] if provider.provider_type == "databento" else
                     [provider.account_id_env, "OANDA_ACCESS_TOKEN"] if provider.provider_type == "oanda" else [])
        providers.append({"provider_id": identity, "type": provider.provider_type,
                          "required_environment_present": {name: bool(os.getenv(name)) for name in variables},
                          "network_checked": False})
    return {"schema_version": 1, "python": platform.python_version(), "platform": platform.platform(),
            "supported_release_python": sys.version_info[:2] == (3, 11),
            "dependencies": versions, "configuration_valid": True,
            "configured_instruments": len(config.instruments), "providers": providers,
            "execution_mode": "local_paper", "live_acceptance": "pending",
            "credentials_included": False}


def _write(result, output=None):
    text = json.dumps(sanitize(result), indent=2, sort_keys=True, allow_nan=False)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _decimal_argument(value: str) -> Decimal:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("expected a finite decimal") from exc
    if not number.is_finite():
        raise argparse.ArgumentTypeError("expected a finite decimal")
    return number


def _fees(args: argparse.Namespace) -> dict[str, Any]:
    """Price an explicit research ticket without opening an account or provider."""
    from .broker_fees import (
        AS_OF,
        FeeAssumptions,
        break_even_gain,
        manual_fx_fee,
        round_trip_fees,
    )

    assumptions = FeeAssumptions(
        args.pricing, args.stamp_rate, args.finra_pass_through == "listed",
        args.tiered_cap, args.monthly_shares_before,
    )
    costs = round_trip_fees(
        args.quantity, args.price, args.price, assumptions,
        entry_liquidity=args.entry_liquidity, exit_liquidity=args.exit_liquidity,
    )
    gain = break_even_gain(
        args.quantity, args.price, assumptions, extra_usd=args.extra_usd,
        entry_liquidity=args.entry_liquidity, exit_liquidity=args.exit_liquidity,
    )
    notional = args.quantity * args.price
    result: dict[str, Any] = {
        "source_snapshot": AS_OF.isoformat(), "pricing": args.pricing,
        "quantity": args.quantity, "entry_price_usd": str(args.price),
        "entry_notional_usd": str(notional),
        "components_usd": {name: str(value) for name, value in costs.components.items()},
        "flat_price_cost_usd": str(costs.total),
        "flat_price_cost_bps": str(costs.total / notional * 10000),
        "exact_nonnegative_break_even_gain_usd": str(gain),
        "exact_nonnegative_break_even_bps": str(gain / notional * 10000),
        "extra_required_usd": str(args.extra_usd),
        "fx_prefunding": None,
        "private_account_access": False, "provider_connections": False,
        "qualification": "unrounded research scenario, not a broker invoice or profit estimate",
        "assumptions": {
            "portuguese_commission_tax_rate": str(args.stamp_rate),
            "portuguese_collection_and_taxable_base": "unconfirmed sensitivity",
            "finra_q4_multiplier": args.finra_pass_through,
            "tiered_cap": str(args.tiered_cap) if args.tiered_cap is not None else "conflict fenced",
            "monthly_shares_before": args.monthly_shares_before,
            "venue_proxy": "Nasdaq-listed displayed Nasdaq execution, not guaranteed SmartRouting",
            "entry_liquidity": args.entry_liquidity, "exit_liquidity": args.exit_liquidity,
            "rounding": "unrounded; customer rounding unconfirmed",
        },
    }
    if args.prefund_usd:
        if args.eur_per_usd is None:
            raise ValueError("--eur-per-usd is required for a prefunding example")
        conversion = manual_fx_fee(
            args.prefund_usd, args.eur_per_usd, stamp_rate=args.stamp_rate,
        )
        result["fx_prefunding"] = {name: str(value) for name, value in conversion.items()}
        result["fx_prefunding"]["cost_scope"] = "one conversion, charged in EUR, not every stock round trip"
    return result


async def _record(args):
    from launch import create_trading_terminal_html

    from .cross_asset_app import run_cross_asset_system
    root = Path(__file__).parents[2]
    directory = Path(args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    factory = lambda port: create_trading_terminal_html(port, directory / "terminal.html")
    task = run_cross_asset_system(str(Path(args.config).resolve()), args.port, args.open_browser,
                                 factory, root, recording_directory=directory)
    if args.duration is None:
        await task
    else:
        try:
            await asyncio.wait_for(task, timeout=args.duration)
        except asyncio.TimeoutError:
            print("Recording duration reached; paper process shut down through its cleanup path.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local paper trading research tools; no exchange execution.")
    sub = parser.add_subparsers(dest="command", required=True)
    fees = sub.add_parser("fees", help="Offline IBIE stock fee and break-even research scenario")
    fees.add_argument("--pricing", required=True, choices=("fixed", "tiered"))
    fees.add_argument("--quantity", required=True, type=int)
    fees.add_argument("--price", required=True, type=_decimal_argument)
    fees.add_argument("--stamp-rate", required=True, type=_decimal_argument,
                      help="explicit unconfirmed Portuguese commission-tax scenario: 0 or 0.04")
    fees.add_argument("--finra-pass-through", required=True, choices=("zero", "listed"),
                      help="explicit Q4 commission-multiplier interpretation")
    fees.add_argument("--tiered-cap", type=_decimal_argument,
                      help="explicit 0.005 or 0.01 branch for cap-sensitive Tiered tickets")
    fees.add_argument("--monthly-shares-before", type=int, default=0)
    fees.add_argument("--entry-liquidity", choices=("add", "remove"), default="add")
    fees.add_argument("--exit-liquidity", choices=("add", "remove"), default="remove")
    fees.add_argument("--extra-usd", type=_decimal_argument, default=Decimal(0))
    fees.add_argument("--prefund-usd", type=_decimal_argument, default=Decimal(0))
    fees.add_argument("--eur-per-usd", type=_decimal_argument)
    fees.add_argument("--output", help="create a new report, refusing an existing file")
    doctor = sub.add_parser("doctor", help="Validate local configuration and installation without connecting")
    doctor.add_argument("--config", default=str(_default_config()))
    doctor.add_argument("--output")
    fixture = sub.add_parser("fixture", help="Create an explicitly synthetic four-market acceptance recording")
    fixture.add_argument("--config", default=str(_default_config()))
    fixture.add_argument("--directory", required=True)
    fixture.add_argument("--seconds", type=int, default=120)
    replay = sub.add_parser("replay", help="Replay recorded arrival order with a virtual clock")
    replay.add_argument("directory")
    replay.add_argument("--config", default=str(_default_config()))
    replay.add_argument("--speed", type=float, default=0)
    replay.add_argument("--seed", type=int)
    replay.add_argument("--drop-every", type=int, default=0)
    replay.add_argument("--duplicate-every", type=int, default=0)
    replay.add_argument("--source-clock-offset", type=float, default=0)
    replay.add_argument("--delivery-delay", type=float, default=0)
    replay.add_argument("--allow-incomplete", action="store_true")
    replay.add_argument("--research-assumption-override", action="store_true",
                        help="Explicit corrected re-evaluation under changed execution assumptions")
    replay.add_argument("--output")
    evaluate = sub.add_parser("evaluate", help="Chronological 60/20/20 strategy research with conservative scenarios")
    evaluate.add_argument("directory")
    evaluate.add_argument("--config", default=str(_default_config()))
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--output")
    record = sub.add_parser("record", help="Run configured read-only feeds and record the local paper input stream")
    record.add_argument("--config", required=True)
    record.add_argument("--directory", required=True)
    record.add_argument("--duration", type=float)
    record.add_argument("--port", type=int, default=0)
    record.add_argument("--open-browser", action="store_true")
    benchmark = sub.add_parser("benchmark", help="Measure synthetic load through the real account writer")
    benchmark.add_argument("--config", default=str(_default_config()))
    benchmark.add_argument("--instruments", type=int, default=25)
    benchmark.add_argument("--rate", type=int, default=1000)
    benchmark.add_argument("--duration", type=float, default=10)
    benchmark.add_argument("--burst-rate", type=int, default=5000)
    benchmark.add_argument("--burst-seconds", type=float, default=30)
    benchmark.add_argument("--soak-seconds", type=float, default=0)
    benchmark.add_argument("--output")
    causal = sub.add_parser("causal", help="Advisory lead/lag analysis of a validated recording; no account writes")
    causal.add_argument("directory")
    causal.add_argument("--output")
    causal.add_argument("--state-out", help="Create a new, separate research-state file (no overwrite)")
    causal.add_argument("--state-in", help="Continue research from a versioned state on a later recording")
    causal.add_argument("--evaluate-direction", action="store_true", help="Report chronological next-bucket calibration, not trading P&L")
    causal.add_argument("--research-config", help="Separate version-2 research parameters JSON; never account configuration")
    args = parser.parse_args(argv)
    if args.command == "fees":
        try:
            result = _fees(args)
        except ValueError as exc:
            parser.error(str(exc))
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        _write(result)
        return 0
    if args.command == "causal":
        from .causal_analysis import analyze_recording
        parameters = json.loads(Path(args.research_config).read_text(encoding="utf-8")) if args.research_config else None
        if args.evaluate_direction and (args.state_in or parameters is not None):
            parser.error("direction evaluation uses one frozen default configuration, without restored state")
        result = analyze_recording(args.directory, state_out=args.state_out, state_in=args.state_in,
                                   parameters=parameters)
        if args.evaluate_direction:
            from .causal_analysis import evaluate_direction_recording
            result["direction_evaluation"] = evaluate_direction_recording(args.directory)
        # Research reports must not overwrite existing account/configuration files.
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        _write(result)
        return 0
    if args.command == "doctor":
        _write(diagnostics(args.config), args.output)
    elif args.command == "fixture":
        from .replay_fixtures import create_representative_fixture
        path = create_representative_fixture(args.directory, args.config, seconds=args.seconds)
        _write({"directory": str(path), "source_kind": "synthetic_fixture"})
    elif args.command == "replay":
        from .replay import FaultSpec, ReplayRunner
        result = ReplayRunner(args.config, seed=args.seed,
                              research_assumption_override=args.research_assumption_override).run(args.directory, speed=args.speed,
            faults=FaultSpec(args.drop_every, args.duplicate_every, args.source_clock_offset, args.delivery_delay),
            allow_incomplete=args.allow_incomplete)
        _write(result, args.output)
    elif args.command == "evaluate":
        from .research_evaluation import evaluate_recording
        _write(evaluate_recording(args.directory, args.config, seed=args.seed), args.output)
    elif args.command == "record":
        if args.duration is not None and args.duration <= 0:
            parser.error("--duration must be positive")
        asyncio.run(_record(args))
    elif args.command == "benchmark":
        from .benchmark import run_benchmark
        _write(run_benchmark(instrument_count=args.instruments, event_rate=args.rate,
            sustained_seconds=args.duration, burst_rate=args.burst_rate,
            burst_seconds=args.burst_seconds, soak_seconds=args.soak_seconds,
            config_path=args.config), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
