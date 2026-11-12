import asyncio
import json
from pathlib import Path
import threading
import time

import pytest
import websockets

from src.config.cross_asset_config import load_cross_asset_config
from src.connectors.market_data import ProviderCapabilities, ProviderState, ProviderStatus
from src.runtime.application_service import PaperApplicationService
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.provider_supervisor import ProviderSupervisor, safe_provider_error
from src.runtime.state_manager import StateManager
from src.runtime.websocket_ui_server import WebSocketUIServer


CONFIG = Path(__file__).parents[1] / "config" / "cross_asset_paper.example.json"


def eventually(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("runtime condition did not become true")


class Provider:
    capabilities = ProviderCapabilities(quote=True, trades=True)

    def __init__(self):
        self.status = ProviderStatus("fixture", ProviderState.STOPPED)
        self.stopped = threading.Event()

    def start(self, handler):
        self.status = ProviderStatus("fixture", ProviderState.CONNECTED)
        handler("fixture-event")

    def stop(self):
        self.status = ProviderStatus("fixture", ProviderState.STOPPED)
        self.stopped.set()


def test_failed_provider_does_not_stop_healthy_feed_or_retry_access_without_request():
    healthy = Provider()
    events = []
    attempts = []

    def unavailable():
        attempts.append(1)
        raise RuntimeError("403 entitlement denied")

    good = ProviderSupervisor("good", ("A",), lambda: healthy,
        lambda *args: events.append(args), lambda *args: None, lambda *args: None, poll_seconds=0.01)
    bad = ProviderSupervisor("bad", ("B",), unavailable,
        lambda *args: None, lambda *args: None, lambda *args: None, poll_seconds=0.01)
    try:
        good.start()
        bad.start()
        eventually(lambda: events and bad.status["state"] == "access_denied")
        assert good.status["state"] == "healthy"
        assert len(attempts) == 1
        assert not healthy.stopped.is_set()
        bad.retry_access()
        eventually(lambda: len(attempts) == 2)
    finally:
        good.stop()
        bad.stop()


def test_provider_messages_redact_account_urls_and_bearer_credentials():
    message = safe_provider_error("GET /accounts/123-456/pricing Bearer abcdef token=secret")
    assert "123-456" not in message
    assert "abcdef" not in message
    assert "secret" not in message


def command(service, request_id, name, payload=None, revision=None):
    return {"protocol_version": 1, "request_id": request_id,
            "account_id": service.account_id,
            "expected_revision": service._revision if revision is None else revision,
            "command": name, "payload": payload or {}}


def test_commands_are_persistently_idempotent_and_stop_ignores_stale_revision(tmp_path):
    manager = StateManager(str(tmp_path / "account.db"))
    engine = CrossAssetPaperEngine(load_cross_asset_config(CONFIG), state_manager=manager)
    service = PaperApplicationService(engine, manager)
    service.start()
    request = command(service, "set-spread", "update_params", {
        "instrument_id": "NASDAQ:AAPL", "spread_bps": 17})
    try:
        result = service.submit_command(request).result(3)
        assert result["success"] is True
        assert service.submit_command(request).result(3) == result
        assert engine._runtime["NASDAQ:AAPL"].strategy.spread_bps == pytest.approx(17)
        changed = dict(request, payload={"instrument_id": "NASDAQ:AAPL", "spread_bps": 19})
        with pytest.raises(ValueError, match="reused|different|conflict"):
            service.submit_command(changed).result(3)
        service.call(lambda: setattr(engine, "is_running", True))
        stopped = service.submit_command(command(service, "emergency-stop", "stop_strategy", revision=0)).result(3)
        assert stopped["success"] is True
        assert engine.is_running is False
    finally:
        service.stop()
    # A lost acknowledgment can be retried after restarting the account.
    restored = CrossAssetPaperEngine(load_cross_asset_config(CONFIG), state_manager=manager)
    restarted = PaperApplicationService(restored, manager)
    restarted.start()
    try:
        assert restarted.submit_command(request).result(3) == result
    finally:
        restarted.stop()


class MinimalEngine:
    account_id = "runtime-fixture"
    state_manager = None
    circuit_breaker_active = False
    is_running = False

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.threads = []
        self.events = []
        self.invalidations = []

    def on(self, event, callback):
        pass

    def get_dashboard_state(self):
        return {"instruments": [], "portfolio": {}, "is_running": self.is_running}

    def get_status(self):
        return {"equity": 1000, "total_pnl": 0}

    def on_market_event(self, event, provider_id=None, generation=None):
        self.threads.append(threading.get_ident())
        self.events.append((event, generation))
        if event == "block":
            self.started.set()
            self.release.wait(3)
        return True

    def invalidate_provider(self, *args):
        self.invalidations.append(args)

    def stop_strategy(self):
        self.is_running = False


def test_queue_overflow_marks_generation_incomplete_and_never_reuses_its_events():
    engine = MinimalEngine()
    service = PaperApplicationService(engine, queue_capacity=1)
    resync = []
    service.on_overflow = lambda *args: resync.append(args)
    service.start()
    try:
        assert service.enqueue_market("feed", "block", 1)
        assert engine.started.wait(2)
        assert service.enqueue_market("feed", "queued-before-overflow", 1)
        assert not service.enqueue_market("feed", "overflow", 1)
        engine.release.set()
        eventually(lambda: engine.invalidations)
        service.call(lambda: None)
        assert engine.events == [("block", 1)]
        assert resync == [("feed", 1)]
        assert service.enqueue_market("feed", "fresh-generation", 2)
        service.call(lambda: None)
        assert engine.events[-1] == ("fresh-generation", 2)
        assert len(set(engine.threads)) == 1
        assert service.get_dashboard_state()["application"]["recording_complete"] is False
    finally:
        engine.release.set()
        service.stop()


def test_loopback_authentication_precedes_any_account_state_and_rejects_remote_origins():
    async def scenario():
        class Bridge:
            def get_full_status(self):
                return {"equity": 1000}

        server = WebSocketUIServer(Bridge(), port=0, session_token="local-session-token")
        await server.start()
        uri = "ws://127.0.0.1:{}".format(server.port)
        try:
            async with websockets.connect(uri, origin="https://untrusted.example") as socket:
                with pytest.raises(websockets.ConnectionClosed):
                    await socket.recv()
            async with websockets.connect(uri, origin="null") as socket:
                await socket.send(json.dumps({"type": "authenticate", "token": "wrong"}))
                with pytest.raises(websockets.ConnectionClosed):
                    await socket.recv()
            async with websockets.connect(uri, origin="null") as socket:
                await socket.send(json.dumps({"type": "authenticate", "token": "local-session-token"}))
                assert json.loads(await socket.recv())["type"] == "authenticated"
                assert json.loads(await socket.recv())["type"] == "full_state"
        finally:
            await server.stop()
    asyncio.run(asyncio.wait_for(scenario(), timeout=8))
