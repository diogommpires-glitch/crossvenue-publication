"""Exercise the real startup boundary with a restored synthetic account."""

import asyncio
import json
from contextlib import suppress
from pathlib import Path

import websockets

from src.config.cross_asset_config import load_cross_asset_config
from src.runtime import cross_asset_app
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.recording import load_manifest
from src.runtime.replay import ReplayRunner
from src.runtime.state_manager import StateManager
from src.runtime.websocket_ui_server import WebSocketUIServer

ROOT = Path(__file__).parents[1]


def test_restored_startup_records_prior_ids_and_releases_writer_on_shutdown(tmp_path, monkeypatch):
    document = json.loads((ROOT / "config/cross_asset_paper.example.json").read_text())
    document["state_db_path"] = str(tmp_path / "account.db")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(document))
    config = load_cross_asset_config(config_path)
    manager = StateManager(config.state_db_path)
    engine = CrossAssetPaperEngine(config, state_manager=manager)
    assert engine.record_funding("prior-funding", "USD", 100, "synthetic")
    expected_cash = engine.ledger.cash_by_currency
    recording = tmp_path / "recording"
    runtime_root = tmp_path / "runtime"
    # The real runner may create daily backups before the test cancels it.
    for relative in ("assets/terminal.html", "pyproject.toml", "requirements-paper.lock"):
        target = runtime_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    servers = []

    def server_factory(**kwargs):
        server = WebSocketUIServer(**kwargs)
        servers.append(server)
        return server

    def html_factory(_port):
        target = tmp_path / "dashboard.html"
        target.write_text("<script>const PAPER_SESSION_TOKEN = '';</script>")
        return str(target)

    monkeypatch.setattr(cross_asset_app, "build_market_data_provider_factories", lambda _config: [])
    monkeypatch.setattr(cross_asset_app, "WebSocketUIServer", server_factory)

    async def scenario():
        task = asyncio.create_task(
            cross_asset_app.run_cross_asset_system(
                str(config_path), 0, False, html_factory, runtime_root, recording
            )
        )
        try:
            for _ in range(200):
                if task.done():
                    await task
                if servers and servers[0].running and (tmp_path / "dashboard.html").exists():
                    break
                await asyncio.sleep(0.01)
            assert servers and servers[0].running
            server = servers[0]
            assert not server.bridge.engine.is_running
            async with websockets.connect(f"ws://127.0.0.1:{server.port}", origin="null") as socket:
                await socket.send(
                    json.dumps({"type": "authenticate", "token": server.session_token})
                )
                assert json.loads(await socket.recv())["type"] == "authenticated"
                assert json.loads(await socket.recv())["type"] == "full_state"
                await socket.send(
                    json.dumps(
                        {
                            "type": "paper_command",
                            "data": {
                                "protocol_version": 1,
                                "request_id": "retry-prior-funding",
                                "account_id": config.account_id,
                                "expected_revision": server.bridge._revision,
                                "command": "record_funding",
                                "payload": {
                                    "funding_id": "prior-funding",
                                    "currency": "USD",
                                    "amount": 100,
                                    "source": "synthetic",
                                },
                            },
                        }
                    )
                )
                while True:
                    result = json.loads(await socket.recv())
                    if result["type"] == "control_response":
                        break
                assert result["success"] and result["data"] is False
                assert server.bridge.engine.ledger.cash_by_currency == expected_cash
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        assert not servers[0].running

    asyncio.run(asyncio.wait_for(scenario(), timeout=10))
    manifest = load_manifest(recording)
    assert manifest["initial_signature_history"]["records"] >= 1
    assert ReplayRunner(config).run(recording)["financial"]["cash"] == expected_cash
    recovered = StateManager(config.state_db_path)
    with recovered.acquire_account_writer(config.account_id):
        restored = CrossAssetPaperEngine(config, state_manager=recovered)
        assert restored.ledger.cash_by_currency == expected_cash
