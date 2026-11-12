"""Application runner for config-driven cross-asset local paper trading."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..config.cross_asset_config import load_cross_asset_config
from ..connectors.provider_factory import build_market_data_provider_factories
from .application_service import PaperApplicationService
from .cross_asset_paper_engine import CrossAssetPaperEngine
from .provider_supervisor import ProviderSupervisor
from .state_manager import StateManager
from .websocket_ui_server import WebSocketUIServer


async def run_cross_asset_system(config_path: str, port: int, open_browser: bool,
                                 html_factory: Callable[[int], str],
                                 project_root: Path, recording_directory=None) -> None:
    """Start the data-only providers, local ledger, and shared terminal UI."""
    config = load_cross_asset_config(config_path)
    database = Path(config.state_db_path)
    if not database.is_absolute():
        database = project_root / database
    state_manager = StateManager(str(database))
    with state_manager.acquire_account_writer(config.account_id):
        await _run_locked_system(config_path, port, open_browser, html_factory,
                                 project_root, recording_directory, config, state_manager)


async def _run_locked_system(config_path, port, open_browser, html_factory,
                             project_root, recording_directory, config, state_manager):
    """The writer lock covers recovery, migrations, and every account mutation."""
    database = state_manager.db_path
    from .replay import DeterministicIds
    session_seed = secrets.randbits(64)
    engine = CrossAssetPaperEngine(config, state_manager=state_manager,
                                  id_factory=DeterministicIds(session_seed))
    from .portfolio_analytics import PortfolioAnalytics
    analytics = PortfolioAnalytics(state_manager=state_manager, account_id=config.account_id)
    recorder = None
    if recording_directory is not None:
        from .recording import SessionRecorder
        code_digest = hashlib.sha256()
        code_files = sorted((project_root / "src").rglob("*.py")) + [
            project_root / "assets/terminal.html", project_root / "pyproject.toml"]
        for source in code_files:
            code_digest.update(source.relative_to(project_root).as_posix().encode())
            code_digest.update(source.read_bytes())
        dependency_hash = hashlib.sha256((project_root / "requirements-paper.lock").read_bytes()).hexdigest()
        recorder = SessionRecorder(recording_directory,
            config=json.loads(Path(config_path).read_text(encoding="utf-8")), seed=session_seed,
            code_version=code_digest.hexdigest(), dependency_hash=dependency_hash,
            source_kind="observed_provider_data")
    bindings = build_market_data_provider_factories(config)
    if recorder is not None:
        recorder.set_initial_state(engine._checkpoint_payload(),
            provider_bindings={provider_id: list(ids) for provider_id, _, ids in bindings},
            signature_history=state_manager.iter_cross_asset_signatures(config.account_id))
    from .causal_analysis import CausalVenueScout
    service = PaperApplicationService(engine, state_manager, recorder=recorder, analytics=analytics,
                                     research=CausalVenueScout(max_streams=8, max_buckets=120))
    supervisors = {}
    for provider_id, factory, instrument_ids in bindings:
        supervisors[provider_id] = ProviderSupervisor(provider_id, instrument_ids, factory,
            service.enqueue_market, service.bind_provider, service.invalidate_provider,
            on_status=service.enqueue_provider_status)
    service.provider_status = lambda: [supervisor.status for supervisor in supervisors.values()]
    service.on_overflow = lambda provider_id, generation: supervisors[provider_id].request_resync()
    session_token = secrets.token_urlsafe(32)
    ui_server = WebSocketUIServer(bridge=service, port=port, application_service=service,
                                 session_token=session_token)
    loop = asyncio.get_running_loop()
    snapshot_broadcast_pending = False

    def schedule_snapshot_broadcast():
        """Coalesce fast feed callbacks into a bounded dashboard update rate."""
        nonlocal snapshot_broadcast_pending
        if snapshot_broadcast_pending or not ui_server.running:
            return
        snapshot_broadcast_pending = True

        def publish_latest_snapshot():
            nonlocal snapshot_broadcast_pending
            snapshot_broadcast_pending = False
            if ui_server.running:
                asyncio.create_task(ui_server.broadcast_update(
                    "multi_asset_state", service.get_dashboard_state()))

        loop.call_later(0.2, publish_latest_snapshot)

    def forward(event_name):
        def callback(payload):
            if event_name == "multi_asset_state":
                if not loop.is_closed():
                    loop.call_soon_threadsafe(schedule_snapshot_broadcast)
                return

            def schedule():
                asyncio.create_task(ui_server.broadcast_update(event_name, payload))
            if not loop.is_closed():
                loop.call_soon_threadsafe(schedule)
        return callback

    for event_name in (
        "multi_asset_state", "strategy_status", "strategy_fill",
        "order_accepted", "order_rejected",
        "orders_cancelled", "provider_error", "persistence_error", "futures_settlement",
        "corporate_action", "fx_financing",
    ):
        engine.on(event_name, forward(event_name))

    html_path = None
    try:
        service.start()
        await ui_server.start_server()
        port = ui_server.port
        html_path = html_factory(port)
        # The token is scoped to this local instance and only written into the
        # generated dashboard, never source config, logs, or recordings.
        html_file = Path(html_path)
        document = html_file.read_text(encoding="utf-8")
        token_marker = "const PAPER_SESSION_TOKEN = '';"
        if token_marker not in document:
            raise RuntimeError("Dashboard session marker is missing; launch with the matching UI template")
        html_file.write_text(document.replace(token_marker,
            "const PAPER_SESSION_TOKEN = {};".format(json.dumps(session_token)), 1), encoding="utf-8")
        for supervisor in supervisors.values():
            supervisor.start()
        print("[PAPER] Cross-asset mode: orders and fills are simulated locally.")
        print("[PAPER] Binance, Databento, and OANDA connections are market-data-only.")
        print("[PAPER] Configured instruments: {}".format(
            ", ".join(item.spec.symbol for item in config.instruments.values())))
        print("[PAPER] Account history: {}".format(database))

        if open_browser:
            webbrowser.open(Path(html_path).resolve().as_uri())
        else:
            print("[UI] Open manually: {}".format(Path(html_path).resolve().as_uri()))
        print("[UI] WebSocket: ws://127.0.0.1:{}".format(port))
        print("[PAPER] Press Ctrl+C to stop; working paper quotes will be cancelled.")

        last_tick = 0.0
        backup_day = None
        while True:
            await asyncio.sleep(0.2)
            now = time.monotonic()
            if now - last_tick >= 1.0:
                service.tick()
                last_tick = now
                current_day = datetime.now(timezone.utc).date()
                if current_day != backup_day:
                    await asyncio.to_thread(service.call, lambda: state_manager.run_daily_backup(
                        directory=project_root / "data" / "backups"))
                    backup_day = current_day
            await ui_server.broadcast_update("multi_asset_state", service.get_dashboard_state())
    except KeyboardInterrupt:
        print("\n[PAPER] Graceful shutdown; the account checkpoint and fill history will be kept.")
    finally:
        try:
            for supervisor in supervisors.values():
                await asyncio.to_thread(supervisor.stop)
            await asyncio.to_thread(service.stop)
        finally:
            await ui_server.stop_server()

    status = service.get_status()
    print("[PAPER] Final equity: {} {:.2f} · total P&L: {:.2f} · fills: {}".format(
        status.get("reporting_currency", config.reporting_currency), status["equity"],
        status["total_pnl"], status["total_fills"]))
