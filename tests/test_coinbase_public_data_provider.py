"""Public quote parsing, timestamp provenance and bounded adapter lifecycle."""

import json
import threading
from dataclasses import replace
from datetime import UTC, datetime

import pytest
import requests

from src.connectors import coinbase_provider
from src.connectors.coinbase_provider import CoinbasePublicQuoteProvider
from src.connectors.market_data import MarketDataProvider, ProviderState
from src.market.events import MarketEventKind
from src.market.instruments import AssetClass, FXAccountingMode, InstrumentSpec
from src.runtime.provider_supervisor import ProviderSupervisor

STAMP = datetime(2026, 11, 11, 12, tzinfo=UTC).timestamp()


def instrument():
    return InstrumentSpec(
        "COINBASE:USDT_USD",
        AssetClass.FX,
        "USDT-USD",
        "COINBASE",
        "USDT",
        "USD",
        "USD",
        "USDT",
        0.000001,
        0.000001,
        0.000001,
        fx_accounting_mode=FXAccountingMode.DELIVERABLE,
    )


def quote(**changes):
    return {
        "bids": [[".9998", "125.5", 2]],
        "asks": [["1.0000", "200", 3]],
        "sequence": 42,
        "auction_mode": False,
        "time": "2026-11-11T12:00:00.123456789Z",
        **changes,
    }


class Response:
    def __init__(self, payload=None, *, status=200, raw=None):
        self.status_code = status
        self.raw_body = (
            json.dumps(payload if payload is not None else quote()).encode() if raw is None else raw
        )
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("public quote request failed")

    def iter_content(self, chunk_size):
        yield self.raw_body

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = threading.Event()

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def close(self):
        self.closed.set()


def provider(**kwargs):
    return CoinbasePublicQuoteProvider(instrument(), receive_clock=lambda: STAMP + 1, **kwargs)


def test_normalizes_real_publisher_time_sequence_and_read_only_quote_contract():
    adapter = provider()
    event = adapter.normalize_message(quote())
    assert isinstance(adapter, MarketDataProvider)
    assert adapter.capabilities.quote and adapter.capabilities.order_book
    assert (
        not adapter.capabilities.trades
        and not adapter.capabilities.sessions
        and not adapter.capabilities.status
    )
    assert event.kind == MarketEventKind.BOOK_SNAPSHOT
    assert event.instrument_id == "COINBASE:USDT_USD" and event.source == "coinbase"
    assert event.event_ts == STAMP + 0.123456789
    assert event.receive_ts == STAMP + 1 and event.sequence == 42
    assert event.bids == ((0.9998, 125.5),) and event.asks == ((1, 200),)
    for name in ("place_order", "submit_order", "cancel_order", "get_account"):
        assert not hasattr(adapter, name)


@pytest.mark.parametrize(
    "changes",
    [
        {"sequence": True},
        {"sequence": "42"},
        {"sequence": 42.0},
        {"sequence": -1},
        {"auction_mode": True},
        {"auction_mode": 0},
        {"auction_mode": None},
        {"time": None},
        {"time": "2026-11-11T12:00:00"},
        {"time": "not-a-time"},
        {"time": "2026-11-11T12:00:05Z"},
        {"time": "2026-11-11T11:59:47Z"},
        {"bids": []},
        {"asks": []},
        {"bids": [["nan", "2", 1]]},
        {"bids": [["1.1", "2", 1]]},
        {"asks": [["1", "0", 1]]},
        {"bids": [[True, "2", 1]]},
        {"asks": [["inf", "2", 1]]},
        {"bids": [["1", "2"]]},
        {"bids": [[".9", "2", 1], [".8", "2", 1]]},
    ],
)
def test_rejects_invalid_auction_stale_future_or_malformed_snapshots(changes):
    with pytest.raises(ValueError):
        provider().normalize_message(quote(**changes))


def test_timestamp_bounds_keep_source_age_and_clock_skew_separate():
    adapter = provider()
    assert adapter.normalize_message(quote(time="2026-11-11T11:59:48Z")).event_ts == STAMP - 12
    assert adapter.normalize_message(quote(time="2026-11-11T12:00:04Z")).event_ts == STAMP + 4
    assert adapter.normalize_message(quote(time="2026-11-11T13:00:00+01:00")).event_ts == STAMP


def test_invalid_timezone_minutes_cannot_be_normalized_into_a_valid_publisher_time():
    with pytest.raises(ValueError, match="timezone"):
        provider().normalize_message(quote(time="2026-11-11T13:00:00+00:60"))


@pytest.mark.parametrize(
    "changes",
    [
        {"base_currency": "EUR"},
        {"quote_currency": "EUR"},
        {"fx_accounting_mode": FXAccountingMode.ROLLING},
    ],
)
def test_rejects_non_deliverable_usdt_usd_instrument(changes):
    with pytest.raises(ValueError):
        CoinbasePublicQuoteProvider(replace(instrument(), **changes))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"symbol": "BTC-USD"},
        {"poll_seconds": 0},
        {"poll_seconds": True},
        {"max_source_age_seconds": float("nan")},
        {"max_clock_skew_seconds": -1},
        {"poll_seconds": "1"},
    ],
)
def test_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        provider(**kwargs)


def test_fetch_uses_only_fixed_public_route_with_bounded_transport_and_closes_response():
    response = Response()
    session = Session([response])
    assert provider(session=session).fetch_quote().sequence == 42
    url, options = session.calls[0]
    assert url == "https://api.exchange.coinbase.com/products/USDT-USD/book"
    assert options["params"] == {"level": 1}
    assert options["timeout"] == (3.0, 3.0)
    assert options["allow_redirects"] is False and options["stream"] is True
    assert "auth" not in options and "Authorization" not in options.get("headers", {})
    assert response.closed and not session.closed.is_set()


def test_invalid_receive_time_cannot_replace_source_provenance():
    with pytest.raises(ValueError):
        provider().normalize_message(quote(), receive_ts="now")


def test_response_body_deadline_closes_response(monkeypatch):
    response = Response()
    samples = iter([0.0, 7.0])
    monkeypatch.setattr(coinbase_provider.time, "monotonic", lambda: next(samples))
    with pytest.raises(requests.Timeout, match="deadline"):
        provider(session=Session([response])).fetch_quote()
    assert response.closed


@pytest.mark.parametrize(
    "response", [Response(status=503), Response(status=302), Response(raw=b"x" * 65537)]
)
def test_http_failure_redirect_or_oversized_body_cannot_become_a_quote(response):
    with pytest.raises((ValueError, requests.RequestException)):
        provider(session=Session([response])).fetch_quote()
    assert response.closed


def test_worker_does_not_publish_duplicate_sequence_or_regressed_source_time():
    session = Session(
        [
            Response(quote(sequence=42)),
            Response(quote(sequence=42)),
            Response(quote(sequence=41)),
            Response(quote(sequence=43, time="2026-11-11T11:59:59Z")),
            Response(quote(sequence=44, time="2026-11-11T12:00:00.5Z")),
        ]
    )
    adapter = provider(session=session, poll_seconds=0.01)
    done = threading.Event()
    events = []

    def receive(event):
        events.append(event)
        if len(events) == 2:
            done.set()

    adapter.start(receive)
    try:
        assert done.wait(2)
    finally:
        adapter.stop()
    assert [event.sequence for event in events] == [42, 44]
    assert adapter.status.events_received == 2 and adapter.status.state == ProviderState.STOPPED
    assert all(event.kind == MarketEventKind.BOOK_SNAPSHOT for event in events)
    assert not session.closed.is_set()


def test_stop_fences_inflight_response_and_restart_waits_for_old_worker():
    entered, release = threading.Event(), threading.Event()
    response = Response()

    class BlockingSession(Session):
        def get(self, url, **kwargs):
            entered.set()
            assert release.wait(5)
            return response

    session = BlockingSession([response])
    adapter = provider(session=session)
    events = []
    adapter.start(events.append)
    assert entered.wait(1)
    adapter.stop()
    with pytest.raises(RuntimeError, match="stopping"):
        adapter.start(events.append)
    release.set()
    adapter._worker.join(2)
    assert not adapter._worker.is_alive() and not events
    assert response.closed and adapter.status.state == ProviderState.STOPPED
    assert not session.closed.is_set()


def test_worker_closes_owned_session_and_same_instance_restart_preserves_sequence(monkeypatch):
    session = Session([Response()])
    monkeypatch.setattr(coinbase_provider.requests, "Session", lambda: session)
    adapter = provider(poll_seconds=0.01)
    received = threading.Event()
    events = []
    adapter.start(lambda event: (events.append(event), received.set()))
    assert received.wait(1)
    adapter.stop()
    assert session.closed.is_set()
    adapter.start(events.append)
    try:
        threading.Event().wait(0.05)
    finally:
        adapter.stop()
    assert len(events) == 1


def test_bad_http_payload_sets_error_for_supervisor_without_local_retry_loop():
    session = Session([Response(quote(auction_mode=True))])
    adapter = provider(session=session, poll_seconds=0.01)
    adapter.start(lambda event: pytest.fail("invalid quote published"))
    adapter._worker.join(2)
    assert adapter.status.state == ProviderState.ERROR
    assert len(session.calls) == 1
    adapter.stop()


def test_stop_is_bounded_while_an_already_started_callback_finishes(monkeypatch):
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    session = Session([Response()])
    monkeypatch.setattr(coinbase_provider.requests, "Session", lambda: session)
    adapter = provider(poll_seconds=0.01)
    events = []

    def receive(event):
        events.append(event)
        entered.set()
        assert release.wait(5)
        raise RuntimeError("synthetic handler failure after stop")

    def stop():
        adapter.stop()
        stopped.set()

    adapter.start(receive)
    stopper = threading.Thread(target=stop)
    try:
        assert entered.wait(1)
        stopper.start()
        assert stopped.wait(1.3), "stop waited on the external callback before its bounded join"
        assert adapter.status.state == ProviderState.STOPPED
        assert not session.closed.is_set(), "owned session must outlive its in-flight callback"
        with pytest.raises(RuntimeError, match="stopping"):
            adapter.start(events.append)
    finally:
        release.set()
        if stopper.ident is not None:
            stopper.join(2)
        adapter.stop()
    assert session.closed.wait(1)
    assert len(events) == 1 and adapter.status.events_received == 1
    assert adapter.status.state == ProviderState.STOPPED


def test_excessively_nested_json_is_rejected_and_supervisor_retries():
    response = Response(raw=b'{"unused":' + b"[" * 1100 + b"0" + b"]" * 1100 + b"}")
    adapters, invalidations, events = [], [], []
    received = threading.Event()

    def factory():
        session = Session([response if not adapters else Response()])
        adapter = provider(session=session, poll_seconds=0.01)
        adapters.append(adapter)
        return adapter

    def invalidate(provider_id, message, generation):
        invalidations.append((provider_id, message, generation, adapters[-1].status.state))

    def receive(provider_id, event, generation):
        events.append((provider_id, event, generation))
        received.set()

    supervisor = ProviderSupervisor(
        "coinbase_valuation",
        (instrument().instrument_id,),
        factory,
        receive,
        lambda *_args: None,
        invalidate,
        poll_seconds=0.01,
        retry_cap=0.01,
    )
    supervisor.start()
    try:
        assert received.wait(2), "malformed response left a dead worker without supervisor retry"
        assert len(adapters) == 2 and response.closed
        assert invalidations[0][0] == "coinbase_valuation"
        assert "ValueError" in invalidations[0][1] and "nesting" in invalidations[0][1]
        assert invalidations[0][2:] == (1, ProviderState.ERROR)
        assert len(events) == 1 and events[0][2] == 2
        assert events[0][1].kind == MarketEventKind.BOOK_SNAPSHOT
    finally:
        supervisor.stop()


@pytest.mark.parametrize("mode", ["fetch", "worker"])
def test_owned_session_cannot_attach_ambient_netrc_credentials(monkeypatch, mode):
    session = requests.Session()
    prepared_headers, credential_lookups = [], []

    def synthetic_netrc_lookup(url):
        credential_lookups.append(url)
        return "synthetic-user", "synthetic-password"

    def send(request, **_kwargs):
        prepared_headers.append(dict(request.headers))
        return Response()

    monkeypatch.setattr(requests.sessions, "get_netrc_auth", synthetic_netrc_lookup)
    monkeypatch.setattr(session, "send", send)
    monkeypatch.setattr(coinbase_provider.requests, "Session", lambda: session)
    adapter = provider()
    if mode == "fetch":
        assert adapter.fetch_quote().sequence == 42
    else:
        received = threading.Event()
        adapter.start(lambda _event: received.set())
        try:
            assert received.wait(1)
        finally:
            adapter.stop()
    assert prepared_headers and all("Authorization" not in headers for headers in prepared_headers)
    assert not credential_lookups, "public feed must not look up ambient credentials"


def test_injected_session_environment_policy_remains_caller_owned():
    session = Session([Response()])
    session.trust_env = True
    assert provider(session=session).fetch_quote().sequence == 42
    assert session.trust_env is True and not session.closed.is_set()
