"""Offline regressions for explicit account control and feed lifecycle."""

import asyncio
import contextlib
import json
import re
import time
from pathlib import Path

import pytest
import websockets

import launch
from src.config.cross_asset_config import load_cross_asset_config
from src.connectors.databento_provider import DatabentoMarketDataProvider
from src.runtime.application_service import PaperApplicationService
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.provider_supervisor import ProviderSupervisor
from src.runtime.state_manager import StateManager
from src.runtime.websocket_ui_server import WebSocketUIServer
from tests.test_application_runtime import MinimalEngine
from tests.test_read_only_market_data_providers import FakeDatabentoClient, equity_spec

CONFIG = Path(__file__).parents[1] / "config/cross_asset_paper.example.json"


def eventually(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("runtime condition did not become true")


class ReadyEngine(MinimalEngine):
    def expire_stale_feeds(self, _now):
        pass

    def market_data_is_fresh(self):
        return True

    def start_strategy(self):
        self.is_running = True


def test_fresh_feed_waits_for_explicit_start_and_stop_stays_stopped():
    engine = ReadyEngine()
    service = PaperApplicationService(engine)
    service.start()
    request = {
        "protocol_version": 1,
        "request_id": "explicit-start",
        "account_id": engine.account_id,
        "expected_revision": 0,
        "command": "start_strategy",
        "payload": {},
    }
    try:
        service.tick()
        service.flush()
        assert not engine.is_running
        assert service.submit_command(request).result(3)["success"]
        assert engine.is_running
        stopped = dict(request, request_id="explicit-stop", command="stop_strategy")
        assert service.submit_command(stopped).result(3)["success"]
        service.tick()
        service.flush()
        assert not engine.is_running
    finally:
        service.stop()


def test_legacy_launcher_authenticates_before_state_and_commands(tmp_path, monkeypatch):
    servers = []
    html_path = tmp_path / "terminal.html"

    class Bridge(ReadyEngine):
        def __init__(self, **_kwargs):
            super().__init__()

        def get_full_status(self):
            return self.get_status()

    class Market:
        ws = None

        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    def server_factory(**kwargs):
        server = WebSocketUIServer(**kwargs)
        servers.append(server)
        return server

    monkeypatch.setattr(launch, "LiveTradingBridge", Bridge)
    monkeypatch.setattr(launch, "AsyncBinanceWebSocket", Market)
    monkeypatch.setattr(launch, "WebSocketUIServer", server_factory)
    original_html_factory = launch.create_trading_terminal_html
    monkeypatch.setattr(
        launch,
        "create_trading_terminal_html",
        lambda port: original_html_factory(port, output_path=html_path),
    )
    monkeypatch.setattr(WebSocketUIServer, "start_when_ready", lambda _self: False)

    async def scenario():
        task = asyncio.create_task(launch.run_trading_system(10000, 0, False))
        try:
            for _ in range(100):
                if task.done():
                    await task
                if html_path.exists():
                    break
                await asyncio.sleep(0.01)
            assert html_path.exists()
            server = servers[0]
            assert isinstance(server.session_token, str) and len(server.session_token) >= 32
            embedded = re.search(
                r"const PAPER_SESSION_TOKEN = (.*?);", html_path.read_text(encoding="utf-8")
            )
            assert embedded is not None and json.loads(embedded.group(1)) == server.session_token
            uri = f"ws://127.0.0.1:{server.port}"
            async with websockets.connect(uri, origin="https://untrusted.example") as socket:
                with pytest.raises(websockets.ConnectionClosed):
                    await socket.recv()
            async with websockets.connect(uri, origin="null") as socket:
                await socket.send(json.dumps({"type": "start_strategy"}))
                with pytest.raises(websockets.ConnectionClosed):
                    await socket.recv()
            async with websockets.connect(uri, origin="null") as socket:
                await socket.send(
                    json.dumps({"type": "authenticate", "token": server.session_token})
                )
                assert json.loads(await socket.recv())["type"] == "authenticated"
                assert json.loads(await socket.recv())["type"] == "full_state"
                await socket.send(json.dumps({"type": "start_strategy"}))
                assert json.loads(await socket.recv())["success"]
                assert server.bridge.is_running
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(asyncio.wait_for(scenario(), timeout=8))


def test_databento_transport_disconnect_invalidates_and_retries_new_generation():
    clients, events, invalidations = [], [], []

    class Client(FakeDatabentoClient):
        connected = True

        def is_connected(self):
            return self.connected

    def factory():
        client = Client(
            on_start=lambda callback: callback(
                {
                    "schema": "trades",
                    "instrument_id": 77,
                    "price": 101.25,
                    "size": 1,
                    "side": "B",
                    "err": "ignored field on a normal trade",
                    "ts_event": time.time_ns(),
                }
            )
        )
        clients.append(client)
        return DatabentoMarketDataProvider(
            "XNAS.ITCH",
            {"AAPL": equity_spec()},
            schemas=("trades",),
            instrument_id_map={77: "AAPL"},
            client=client,
        )

    supervisor = ProviderSupervisor(
        "stocks",
        ("NASDAQ:AAPL",),
        factory,
        lambda *args: events.append(args),
        lambda *args: None,
        lambda *args: invalidations.append(args),
        poll_seconds=0.01,
        retry_cap=0.01,
    )
    supervisor.start()
    try:
        eventually(lambda: len(events) == 1)
        clients[0].connected = False
        eventually(lambda: len(events) >= 2)
        assert clients[0].stopped
        assert invalidations[0][0] == "stocks"
        assert invalidations[0][2] == 1
        assert events[0][2] == 1 and events[1][2] == 2
    finally:
        supervisor.stop()


def test_rejected_partial_import_returns_and_caches_committed_revision(tmp_path):
    config = load_cross_asset_config(CONFIG)
    now = 1792497600
    manager = StateManager(str(tmp_path / "account.db"))
    engine = CrossAssetPaperEngine(config, state_manager=manager, clock=lambda: now)
    service = PaperApplicationService(engine, manager, clock=lambda: now)
    action = {
        "action_id": "valid-first",
        "action_type": "financing",
        "instrument_id": "OANDA:EUR_USD",
        "source": "synthetic",
        "effective_ts": now + 100,
        "payload": {"amount": -2, "currency": "USD"},
    }
    request = {
        "protocol_version": 1,
        "request_id": "partial-import",
        "account_id": config.account_id,
        "expected_revision": 0,
        "command": "import_account_actions",
        "payload": {"actions": [action, {}]},
    }
    result = service._execute_command(request)
    assert result["code"] == "rejected"
    assert [item["action_id"] for item in engine.get_account_actions()] == ["valid-first"]
    revision = manager.get_account_revision(config.account_id)
    assert revision > 0 and result["revision"] == revision
    assert service._execute_command(request) == result
    assert manager.get_command_result(config.account_id, "partial-import")["revision"] == revision


def test_databento_error_record_blocks_automatic_access_retry():
    from databento_dbn import ErrorMsg

    clients = []

    def factory():
        client = FakeDatabentoClient(
            on_start=lambda callback: callback(
                ErrorMsg(ts_event=time.time_ns(), err="403 entitlement denied")
            )
        )
        clients.append(client)
        return DatabentoMarketDataProvider("XNAS.ITCH", {"AAPL": equity_spec()}, client=client)

    supervisor = ProviderSupervisor(
        "stocks",
        ("NASDAQ:AAPL",),
        factory,
        lambda *args: None,
        lambda *args: None,
        lambda *args: None,
        poll_seconds=0.01,
        retry_cap=0.01,
    )
    supervisor.start()
    try:
        eventually(lambda: supervisor.status["state"] == "access_denied")
        time.sleep(0.05)
        assert len(clients) == 1
        assert "entitlement denied" in supervisor.status["message"]
    finally:
        supervisor.stop()
