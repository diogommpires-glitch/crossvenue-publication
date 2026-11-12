import sqlite3
import threading
import time

import pytest

from src.runtime.account_store import AccountWriterLock
from src.runtime.application_service import PaperApplicationService
from src.runtime.state_manager import StateManager


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("shutdown condition did not become true")


class SlowEngine:
    account_id = "shutdown-account"
    circuit_breaker_active = False
    is_running = True

    def __init__(self, manager):
        self.state_manager = manager
        self.started = threading.Event()
        self.release = threading.Event()
        self.stopped = threading.Event()

    def on(self, *_):
        pass

    def on_market_event(self, *_args, **_kwargs):
        with self.state_manager._get_connection() as connection:
            self.started.set()
            self.release.wait(5)
            connection.execute("SELECT 1")
        return True

    def stop_strategy(self):
        self.is_running = False
        self.stopped.set()

    def get_dashboard_state(self):
        return {"instruments": [], "portfolio": {}, "is_running": self.is_running}

    def get_status(self):
        return {"equity": 1000, "total_pnl": 0}


def test_expired_stop_call_deadline_still_waits_for_the_writer_and_keeps_lease(tmp_path):
    manager = StateManager(str(tmp_path / "account.db"))
    engine = SlowEngine(manager)
    service = PaperApplicationService(engine, manager)
    returned = threading.Event()
    errors, joins = [], []

    def owner():
        with manager.acquire_account_writer(engine.account_id):
            service.start()
            worker = service._worker
            original_join = worker.join
            # Simulate elapsed shutdown deadlines without making the test
            # sleep for the old ten-second timeout.
            def join(timeout=None):
                joins.append(timeout)
                return original_join(timeout=None if timeout is None else 0.01)
            worker.join = join
            original_call = service.call
            service.call = lambda callback, timeout=None: original_call(callback, timeout=0.01)
            service.enqueue_market("fixture", "slow-input")
            assert engine.started.wait(2)
            try:
                service.stop()
            except TimeoutError as exc:
                errors.append(exc)
            finally:
                returned.set()

    thread = threading.Thread(target=owner)
    thread.start()
    try:
        assert engine.started.wait(2)
        wait_for(lambda: service._shutdown_started)
        time.sleep(0.05)
        assert not returned.is_set()
        assert thread.is_alive()
        with pytest.raises(RuntimeError, match="another process"):
            with AccountWriterLock(manager.db_path, engine.account_id):
                pass
        with pytest.raises(RuntimeError, match="shutting down"):
            service.submit_command({"command": "start_strategy"}).result(1)
    finally:
        engine.release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert engine.stopped.is_set()
    assert returned.is_set() and len(errors) == 1
    assert joins == [None]
    # Repeated shutdown is harmless, and only now may another writer enter.
    service.stop()
    with manager.acquire_account_writer(engine.account_id):
        pass


def test_registry_shutdown_waits_for_active_contexts_and_blocks_new_connections(tmp_path):
    manager = StateManager(str(tmp_path / "account.db"))
    owner_ready = threading.Event()
    leave_scope = threading.Event()
    connection_ready = threading.Event()
    release_connection = threading.Event()
    worker_connections, errors = [], []

    def owner():
        with manager.acquire_account_writer("paper"):
            owner_ready.set()
            leave_scope.wait(5)

    def active_reader():
        try:
            with manager._get_connection() as connection:
                worker_connections.append(connection)
                connection_ready.set()
                release_connection.wait(5)
                assert connection.execute("SELECT 1").fetchone()[0] == 1
        except Exception as exc:
            errors.append(exc)

    owner_thread = threading.Thread(target=owner)
    owner_thread.start()
    assert owner_ready.wait(2)
    worker = threading.Thread(target=active_reader)
    worker.start()
    try:
        assert connection_ready.wait(2)
        leave_scope.set()
        wait_for(lambda: manager._writer_connections_closing)
        assert owner_thread.is_alive()
        with pytest.raises(RuntimeError, match="shutting down"):
            manager.get_account_revision("paper")
    finally:
        release_connection.set()
        leave_scope.set()
        worker.join(5)
        owner_thread.join(5)
    assert not errors
    assert not worker.is_alive() and not owner_thread.is_alive()
    assert manager._writer_connections is None
    assert not manager._writer_connections_closing
    assert manager.get_account_revision("paper") == 0
    with pytest.raises(sqlite3.ProgrammingError):
        worker_connections[0].execute("SELECT 1")
