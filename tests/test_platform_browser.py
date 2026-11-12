"""Actual Chromium + authenticated loopback backend; all prices are fixtures."""

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time

import pytest

playwright = pytest.importorskip("playwright.sync_api")
from launch import create_trading_terminal_html
from src.config.cross_asset_config import load_cross_asset_config
from src.connectors.market_data import ProviderCapabilities
from src.market.events import MarketDataEvent, MarketEventKind
from src.runtime.application_service import PaperApplicationService
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.portfolio_analytics import PortfolioAnalytics
from src.runtime.replay import _ReplayProvider
from src.runtime.state_manager import StateManager
from src.runtime.websocket_ui_server import WebSocketUIServer
from src.utils.types import Side


pytestmark = pytest.mark.browser
ROOT = Path(__file__).parents[1]


class BrowserBackend:
    def __init__(self, directory, unavailable_provider=None):
        self.config = load_cross_asset_config(ROOT / "config" / "cross_asset_paper.example.json")
        self.manager = StateManager(str(directory / "browser.db"))
        self.engine = CrossAssetPaperEngine(self.config, state_manager=self.manager, max_clock_skew_seconds=0)
        self.service = PaperApplicationService(self.engine, self.manager,
            analytics=PortfolioAnalytics(self.manager, self.config.account_id))
        self.service.start()
        for provider_id, provider_config in self.config.providers.items():
            provider = _ReplayProvider(provider_id)
            if provider_config.provider_type == "oanda":
                provider.capabilities = ProviderCapabilities(quote=True, sessions=True, trades=False)
            self.service.bind_provider(provider_id, provider, provider_config.instrument_ids)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.skipped = {unavailable_provider} if unavailable_provider else set()
        if unavailable_provider:
            self.service.provider_status = lambda: [{
                "provider_id": unavailable_provider,
                "instrument_ids": self.config.providers[unavailable_provider].instrument_ids,
                "state": "access_denied",
                "message": "Configure a read-only market-data key and entitlement for this feed.",
            }]
        self.errors = []
        self.mids = {"crypto": 80000.0, "equity": 200.0, "fx": 1.1, "future": 5000.0}
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        assert self.ready.wait(10), "local test backend did not start"
        assert not self.errors, self.errors
        path = Path(create_trading_terminal_html(self.server.port, directory / "dashboard.html"))
        path.write_text(path.read_text(encoding="utf-8").replace(
            "const PAPER_SESSION_TOKEN = '';", "const PAPER_SESSION_TOKEN = 'browser-fixture-token';"), encoding="utf-8")
        self.url = path.resolve().as_uri()

    def run(self):
        try:
            asyncio.run(self.serve())
        except BaseException as exc:
            self.errors.append(str(exc))
            self.ready.set()

    async def serve(self):
        self.loop = asyncio.get_running_loop()
        self.server = WebSocketUIServer(bridge=self.service, port=0,
            application_service=self.service, session_token="browser-fixture-token")
        await self.server.start_server()
        self.ready.set()
        try:
            while not self.stop_event.is_set():
                now = time.time()
                for identity, item in self.config.instruments.items():
                    if item.provider_id in self.skipped:
                        continue
                    mid, tick = self.mids[item.spec.asset_class.value], item.spec.tick_size
                    source = "browser-fixture:" + item.provider_id
                    self.service.enqueue_market(item.provider_id, MarketDataEvent(identity, source,
                        MarketEventKind.STATUS, now, now, status="tradeable"))
                    self.service.enqueue_market(item.provider_id, MarketDataEvent(identity, source,
                        MarketEventKind.BOOK_SNAPSHOT, now, now,
                        bids=((item.spec.round_price(mid - tick, Side.BUY), 100000),),
                        asks=((item.spec.round_price(mid + tick, Side.SELL), 100000),)))
                    if item.fill_model == "trade_print_queue":
                        self.service.enqueue_market(item.provider_id, MarketDataEvent(identity, source,
                            MarketEventKind.TRADE, now, now, price=item.spec.round_price(mid),
                            quantity=1, side=Side.BUY))
                self.service.tick()
                self.service.flush()
                await self.server.broadcast_update("multi_asset_state", self.service.get_dashboard_state())
                await asyncio.sleep(0.15)
        finally:
            await self.server.stop_server()

    def disconnect_clients(self):
        async def close():
            for client in list(self.server.clients):
                await client.close()
        asyncio.run_coroutine_threadsafe(close(), self.loop).result(timeout=5)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        self.service.stop()
        assert not self.thread.is_alive()
        assert not self.errors, self.errors


@contextmanager
def browser_fixture(tmp_path, unavailable_provider=None):
    backend = BrowserBackend(tmp_path, unavailable_provider=unavailable_provider)
    try:
        with playwright.sync_playwright() as driver:
            try:
                browser = driver.chromium.launch(headless=True)
            except Exception as exc:
                pytest.fail("Chromium is required: run python -m playwright install chromium. " + str(exc))
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            external_requests = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            def route(request):
                if request.request.url.startswith(("https://", "http://")):
                    external_requests.append(request.request.url)
                    request.abort()
                else:
                    request.continue_()
            page.route("**/*", route)
            page.goto(backend.url)
            page.wait_for_function("isConnected && state.crossAssetMode && applicationAccountId", timeout=10000)
            playwright.expect(page.locator("#instrumentGrid .instrument-card")).to_have_count(4)
            yield backend, page
            assert not errors, errors
            assert not external_requests, external_requests
            browser.close()
    finally:
        backend.close()


def show_view(page, name):
    page.locator(f'nav [data-view="{name}"]').click()
    playwright.expect(page.locator(f"#view-{name}")).to_be_visible()
    assert page.evaluate("location.hash") == "#" + name


@pytest.mark.parametrize("width", [375, 1440])
def test_read_only_research_filter_evidence_and_reconnect(tmp_path, width):
    with browser_fixture(tmp_path) as (backend, page):
        page.set_viewport_size({"width": width, "height": 900})
        show_view(page, "system")
        playwright.expect(page.locator("#researchStatus")).to_contain_text("disabled")
        freeze_fixture_updates(page)
        page.evaluate("""() => {
            crossAssetState.research = {status:'ready', processing_ts:100, compute_ms:1,
              streams:[{provider:'fixture', generation:2, instrument_id:'A', price_kind:'midpoint',
                stale:false, age_seconds:1, samples:30, positive_return_probability:0.6,
                uncertainty:0.01, regime:'SLOW', entropy:0.9, decay_weight:0.8,
                lead_strength:0.7, research_priority:0.5},
                {provider:'other', generation:1, instrument_id:'B', price_kind:'trade', stale:true, samples:0}],
              edges:[{source:['fixture','A'], target:['other','B'], lag_buckets:1, samples:20, correlation:0.8}],
              diagnostics:{price_policy:4}, skipped_capacity:0};
            renderResearch(crossAssetState.research);
        }""")
        playwright.expect(page.locator("#researchBody tr")).to_have_count(2)
        page.locator("#researchInstrument").select_option("A")
        playwright.expect(page.locator("#researchBody tr")).to_have_count(1)
        playwright.expect(page.locator("#researchBody")).to_contain_text("midpoint")
        page.locator("#researchTitle").locator("..").locator("..").locator("summary").click()
        playwright.expect(page.locator("#researchEdges")).to_contain_text("0.800")
        playwright.expect(page.locator("#researchDiagnostics")).to_contain_text('"price_policy":4')
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator("#researchInstrument").focus()
        assert page.evaluate("document.activeElement.id") == "researchInstrument"
        backend.disconnect_clients()
        page.wait_for_function("!isConnected")
        page.wait_for_function("isConnected", timeout=10000)
        playwright.expect(page.locator("#researchStatus")).to_contain_text("disabled")


def select_market(page, instrument_id):
    if page.locator("#watchlistPanel").get_attribute("open") is None:
        page.locator("#watchlistPanel > summary").click()
    page.locator(f'#instrumentGrid [data-instrument-id="{instrument_id}"]').click()
    page.wait_for_function("identity => selectedInstrumentId === identity", arg=instrument_id)


def freeze_fixture_updates(page):
    # The retained connection is authenticated, but these UI-only cases replace
    # its read model with explicitly synthetic snapshots instead of trading.
    page.evaluate("ws.onmessage = () => {}")


def assert_text_fits_viewport(element):
    geometry = element.evaluate("""element => {
        const box = element.getBoundingClientRect();
        const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
        const rectangles = [];
        while (walker.nextNode()) {
            if (!walker.currentNode.textContent.trim()) continue;
            const range = document.createRange();
            range.selectNodeContents(walker.currentNode);
            rectangles.push(...[...range.getClientRects()].map(rect => ({
                left:rect.left, right:rect.right, top:rect.top, bottom:rect.bottom})));
        }
        return {text:element.textContent.trim(), width:innerWidth,
            box:{left:box.left, right:box.right, top:box.top, bottom:box.bottom}, rectangles};
    }""")
    assert_viewport_geometry(geometry)


def assert_viewport_geometry(geometry):
    assert geometry["rectangles"], geometry
    for rect in geometry["rectangles"]:
        box = geometry["box"]
        assert rect["left"] >= max(0, box["left"]) - 1, geometry
        assert rect["right"] <= min(geometry["width"], box["right"]) + 1, geometry
        assert box["top"] - 1 <= rect["top"] <= rect["bottom"] <= box["bottom"] + 1, geometry


def test_unmeasured_self_impact_has_no_blank_chart(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        show_view(page, "portfolio")
        playwright.expect(page.locator("#currentImpact")).to_be_visible()
        playwright.expect(page.locator("#impactChart")).to_be_hidden()
        assert page.evaluate("Chart.getChart(document.getElementById('impactChart')) === undefined")
        playwright.expect(page.locator("#currentImpact")).to_have_text(
            "Unavailable. Not measured in this runtime.")


def test_mobile_activity_empty_states_are_fully_readable_without_horizontal_scroll(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        freeze_fixture_updates(page)
        page.set_viewport_size({"width": 375, "height": 812})
        show_view(page, "activity")
        playwright.expect(page.locator("#commandLogBody")).to_have_text(
            "No commands in this browser session. Financial history is saved by the backend.")
        for identity in ("commandLogBody", "tradeAttributionBody"):
            cell = page.locator(f"#{identity} td").first
            playwright.expect(cell).to_be_visible()
            assert_text_fits_viewport(cell)
            assert cell.evaluate("""element => {
                const region = element.closest('.scrollable-content');
                return !region || region.scrollWidth <= region.clientWidth;
            }""")


def test_idle_connection_feedback_is_quiet_and_accepted_outcome_survives_reconnect(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        page.wait_for_function("pendingCommands.size === 0")
        playwright.expect(page.locator("#statusText")).to_have_text("Connected to local paper backend")
        playwright.expect(page.locator("#controlMessage")).to_be_hidden()
        page.evaluate("""() => {
            trackCommand('ui-accepted-fixture', 'stop_strategy', {}, false);
            handleMessage({type:'control_response', action:'stop_strategy',
                request_id:'ui-accepted-fixture', success:true, message:'Fixture command acknowledged.'});
        }""")
        playwright.expect(page.locator("#controlMessage")).to_be_visible()
        playwright.expect(page.locator("#controlMessage")).to_have_attribute("data-outcome", "accepted")
        page.evaluate("ws.close()")
        page.wait_for_function("!isConnected")
        page.wait_for_function("isConnected && !waitingForFreshState", timeout=10000)
        playwright.expect(page.locator("#controlMessage")).to_have_attribute("data-outcome", "accepted")
        playwright.expect(page.locator("#controlMessage")).to_contain_text("Fixture command acknowledged.")


@pytest.mark.parametrize("count", [1, 2])
def test_quote_readiness_names_one_or_multiple_working_quotes(tmp_path, count):
    with browser_fixture(tmp_path) as (_, page):
        freeze_fixture_updates(page)
        page.evaluate("""count => {
            const data = structuredClone(crossAssetState);
            data.is_running = true;
            const selected = data.instruments.find(row => row.instrument_id === selectedInstrumentId);
            selected.market_data_ready = true;
            selected.quoting_enabled = true;
            selected.active_orders = Array.from({length:count}, (_, index) => ({
                order_id:`readiness-fixture-${index}`, instrument_id:selected.instrument_id,
                side:'BUY', price:79000, quantity:0.001, remaining_quantity:0.001,
                accepted_at:Date.now()/1000}));
            renderCrossAssetState(data);
        }""", count)
        playwright.expect(page.locator(".readiness-details > summary")).to_have_text("Quote readiness")
        noun = "quote" if count == 1 else "quotes"
        playwright.expect(page.locator("#selectedReadiness")).to_have_text(
            f"{count} working local paper {noun}. Fills require qualifying market events.")


@pytest.mark.parametrize("width", [375, 1440])
def test_portfolio_and_activity_preserve_every_table_value_at_mobile_and_desktop_widths(tmp_path, width):
    with browser_fixture(tmp_path) as (_, page):
        freeze_fixture_updates(page)
        page.set_viewport_size({"width": width, "height": 1000})
        page.evaluate("""() => {
            const data = structuredClone(crossAssetState);
            const equity = data.instruments.find(row => row.instrument_id === 'NASDAQ:AAPL');
            equity.position = 3;
            equity.last_rejection = {timestamp:1793404800, reason:'fixture_limit_exceeded',
                order_id:'synthetic-browser-rejected-order-123456789'};
            data.recent_fills = [{fill_id:'ui-responsive-fill', instrument_id:equity.instrument_id,
                side:'BUY', price:199.25, quantity:3, realized_pnl:1.5, total_pnl:2.5,
                timestamp:1793404800, execution_model:'Synthetic browser fixture'}];
            renderCrossAssetState(data);
            trackCommand('ui-responsive-command', 'stop_strategy', {}, false);
            handleMessage({type:'control_response', action:'stop_strategy',
                request_id:'ui-responsive-command', success:false,
                message:'Fixture account changed; review the latest paper account state before retrying.'});
        }""")
        tables = {
            "portfolio": {
                "positionsBody": ["Instrument", "Position", "Mark", "Unrealized P&L", "Realized P&L", "Valuation", "State"],
                "cashBalancesBody": ["Currency", "Cash", "Available", "Reserved cash", "Reserved margin"],
            },
            "activity": {
                "commandLogBody": ["Time", "Action", "Outcome", "Message"],
                "rejectionLogBody": ["Time", "Instrument", "Reason", "Order"],
                "tradeAttributionBody": ["Time", "Instrument", "Side", "Quantity", "Price", "Realized Δ", "Account P&L at fill"],
            },
        }
        for view, bodies in tables.items():
            show_view(page, view)
            for identity, labels in bodies.items():
                body = page.locator("#" + identity)
                table = body.locator("xpath=ancestor::table")
                assert table.locator("th").all_text_contents() == labels
                assert body.locator("tr").count() > 0, identity
                if width == 375:
                    page.wait_for_function("""({identity, labels}) => {
                        const rows = [...document.getElementById(identity).rows];
                        return rows.every(row => [...row.cells].every((cell, index) =>
                            cell.dataset.label === labels[index]
                            && getComputedStyle(cell, '::before').content === JSON.stringify(labels[index])));
                    }""", arg={"identity": identity, "labels": labels})
                for row_index, row in enumerate(body.locator("tr").all()):
                    cells = row.locator("td")
                    assert cells.count() == len(labels), identity
                    # The command timer replaces rows; measure one current row
                    # in one JS task rather than retaining detached cell handles.
                    geometry = body.evaluate("""(body, rowIndex) => {
                        const row = body.rows[rowIndex];
                        return row ? [...row.cells].map(cell => {
                            const box = cell.getBoundingClientRect();
                            const walker = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
                            const rectangles = [];
                            while (walker.nextNode()) {
                                if (!walker.currentNode.textContent.trim()) continue;
                                const range = document.createRange();
                                range.selectNodeContents(walker.currentNode);
                                rectangles.push(...[...range.getClientRects()].map(rect => ({
                                    left:rect.left, right:rect.right, top:rect.top, bottom:rect.bottom})));
                            }
                            return {text:cell.textContent.trim(), width:innerWidth,
                                display:getComputedStyle(cell).display,
                                box:{x:box.x, y:box.y, width:box.width, height:box.height,
                                    left:box.left, right:box.right, top:box.top, bottom:box.bottom},
                                rectangles};
                        }) : [];
                    }""", row_index)
                    assert len(geometry) == len(labels), (identity, geometry)
                    boxes = [cell["box"] for cell in geometry]
                    assert all(box["height"] > 0 for box in boxes), (identity, boxes)
                    if width == 375:
                        for cell in geometry:
                            assert_viewport_geometry(cell)
                        assert all(second["y"] >= first["y"] + first["height"] - 1
                                   for first, second in zip(boxes, boxes[1:])), (identity, boxes)
                        region = table.locator("xpath=ancestor::div[contains(@class,'scrollable-content')]")
                        assert region.evaluate("element => element.scrollWidth <= element.clientWidth"), identity
                    else:
                        assert table.locator("thead").evaluate("element => getComputedStyle(element).display") == "table-header-group"
                        assert all(cell["display"] == "table-cell" for cell in geometry), (identity, geometry)
                        assert max(box["y"] for box in boxes) - min(box["y"] for box in boxes) <= 1
            if view == "activity":
                playwright.expect(page.locator("#rejectionLogBody")).to_contain_text("synthetic-browser-rejected-order-123456789")
                playwright.expect(page.locator("#tradeAttributionBody")).to_contain_text("199.25 USD")
            else:
                playwright.expect(page.locator("#positionsBody")).to_contain_text("3 shares")
                playwright.expect(page.locator("#cashBalancesBody")).to_contain_text("40,000 USD")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


@pytest.mark.parametrize("width", [375, 1440])
def test_normal_watchlist_rows_are_compact_without_clipping_symbols_or_prices(tmp_path, width):
    with browser_fixture(tmp_path) as (_, page):
        page.set_viewport_size({"width": width, "height": 1000})
        if page.locator("#watchlistPanel").get_attribute("open") is None:
            page.locator("#watchlistPanel > summary").click()
        rows = page.locator("#instrumentGrid .instrument-card")
        playwright.expect(rows).to_have_count(4)
        for row in rows.all():
            box = row.bounding_box()
            assert 44 <= box["height"] <= 52, (row.inner_text(), box)
            assert_text_fits_viewport(row.locator(".instrument-card-title"))
            assert_text_fits_viewport(row.locator(".instrument-price"))


def test_offline_dashboard_instruments_controls_history_and_reconnect(tmp_path):
    with browser_fixture(tmp_path) as (backend, page):
        playwright.expect(page.locator("#multiAssetPanel")).to_be_visible()
        assert page.locator("#controlsPanel").get_attribute("open") is None
        playwright.expect(page.locator("#startButton")).to_be_visible()
        playwright.expect(page.locator("#globalStopButton")).to_be_visible()
        output = Path(os.environ.get("PAPER_BROWSER_EVIDENCE", str(tmp_path)))
        output.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(output / "paper-dashboard-controls-collapsed.png"), full_page=True)
        page.locator("#controlsPanelToggle").focus()
        page.keyboard.press("Enter")
        assert page.locator("#controlsPanel").get_attribute("open") is not None
        for instrument, label in (("NASDAQ:AAPL", "AAPL"), ("CME:ESZ6", "ESZ6"),
                                  ("OANDA:EUR_USD", "EUR/USD"), ("BINANCE:BTCUSDT", "BTC/USDT")):
            select_market(page, instrument)
            playwright.expect(page.locator("#liveBookTitle")).to_contain_text(label)
        page.locator('[data-instrument-id="NASDAQ:AAPL"]').focus()
        page.evaluate("renderCrossAssetState(crossAssetState)")
        playwright.expect(page.locator('[data-instrument-id="NASDAQ:AAPL"]')).to_be_focused()
        select_market(page, "BINANCE:BTCUSDT")
        for identity in ("startButton", "stopButton"):
            box = page.locator("#" + identity).bounding_box()
            assert abs(box["width"] - box["height"]) < 2
        page.locator("#strategyVariables summary").click()
        assert page.locator("#strategyVariables").get_attribute("open") is not None
        page.screenshot(path=str(output / "paper-dashboard-controls-expanded.png"), full_page=True)
        if backend.engine.is_running:
            page.locator("#stopButton").click()
        page.wait_for_function("!strategyState.running")
        assert not backend.engine.get_active_orders_detail()
        manual = backend.service.call(lambda: backend.engine.submit_paper_order(
            "BINANCE:BTCUSDT", Side.BUY, 0.001, 79000.0))
        assert manual["accepted"], manual
        page.wait_for_function("state.activeOrders.length > 0")
        page.locator("#globalStopButton").click()
        page.wait_for_function("state.activeOrders.length === 0")
        assert not backend.engine.get_active_orders_detail()
        page.locator("#startButton").click()
        page.wait_for_function("strategyState.running")
        page.locator("#controlsPanelToggle").focus()
        page.keyboard.press("Space")
        assert page.locator("#controlsPanel").get_attribute("open") is None
        page.locator("#globalStopButton").click()
        page.wait_for_function("!strategyState.running")
        playwright.expect(page.locator("#controlMessage")).to_have_attribute("data-outcome", "accepted")
        assert not backend.engine.get_active_orders_detail()

        started = time.perf_counter()
        backend.mids["crypto"] = 80005
        page.wait_for_function("state.midPrice === 80005", timeout=3000)
        freshness = time.perf_counter() - started
        assert freshness < 1
        show_view(page, "activity")
        page.locator("#loadHistoryButton").click()
        page.wait_for_function("persistedHistoryRows.length > 0", timeout=5000)
        playwright.expect(page.locator("#historyStatus")).to_contain_text("observation")
        backend.disconnect_clients()
        page.wait_for_function("!isConnected", timeout=3000)
        page.wait_for_function("isConnected && state.crossAssetMode", timeout=10000)
        page.reload()
        page.wait_for_function("isConnected && state.crossAssetMode", timeout=10000)
        show_view(page, "activity")
        page.locator("#loadHistoryButton").click()
        page.wait_for_function("persistedHistoryRows.length > 0", timeout=5000)
        page.screenshot(path=str(output / "paper-dashboard.png"), full_page=True)
        (output / "browser_metrics.json").write_text(json.dumps({
            "kind": "actual_chromium_loopback_fixture", "external_network_requests": 0,
            "instrument_count": 4, "dashboard_freshness_seconds": freshness,
            "freshness_under_one_second": freshness < 1,
            "controls_history_reconnect": "passed", "execution_mode": "local_paper"}, indent=2), encoding="utf-8")


def test_missing_futures_access_stays_visible_without_reusing_crypto_price(tmp_path):
    with browser_fixture(tmp_path, unavailable_provider="databento_cme") as (_, page):
        future = page.locator('[data-instrument-id="CME:ESZ6"]')
        playwright.expect(future).to_be_visible()
        playwright.expect(future).to_contain_text("Futures")
        playwright.expect(page.locator("#midPrice")).to_contain_text("80,000")
        select_market(page, "CME:ESZ6")
        playwright.expect(page.locator("#liveBookTitle")).to_contain_text("ESZ6")
        playwright.expect(page.locator("#midPrice")).to_have_text("—")
        playwright.expect(page.locator("#marketSpread")).to_have_text("—")
        playwright.expect(page.locator("#orderBookBody")).to_contain_text("Waiting for this instrument")
        playwright.expect(page.locator("#selectedReadiness")).to_contain_text("read-only market-data key and entitlement")
        playwright.expect(page.locator("#readinessDetails")).to_contain_text("read-only market-data key and entitlement")
        select_market(page, "NASDAQ:AAPL")
        playwright.expect(page.locator("#midPrice")).to_contain_text("200 USD")


def test_sourced_account_adjustment_reconciles_and_survives_browser_reload(tmp_path):
    with browser_fixture(tmp_path) as (backend, page):
        select_market(page, "NASDAQ:AAPL")
        show_view(page, "system")
        page.locator("#paperAdjustmentDetails summary").first.click()
        page.locator("#adjustmentSource").fill("Synthetic browser test entitlement, not provider data")
        page.locator("#dividendPerShare").fill("0.5")
        page.locator("#dividendEligibleQuantity").fill("2")
        page.locator("#dividendActionId").fill("browser-test-dividend")
        page.on("dialog", lambda dialog: dialog.accept())
        page.locator("#applyDividendButton").click()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and backend.engine.ledger.export_state()["income"].get("USD", 0) != 1:
            page.wait_for_timeout(100)
        assert backend.engine.ledger.export_state()["income"]["USD"] == 1
        page.reload()
        page.wait_for_function("isConnected && state.crossAssetMode", timeout=10000)
        recovered = backend.manager.load_cross_asset_snapshot(backend.config.account_id)
        assert recovered["snapshot"]["ledger"]["income"]["USD"] == 1


@pytest.mark.parametrize("viewport", [(375, 812), (768, 1024), (1280, 800), (1440, 900), (1920, 1080)])
def test_workbench_controls_views_and_chart_bounds(tmp_path, viewport):
    width, height = viewport
    with browser_fixture(tmp_path) as (_, page):
        page.set_viewport_size({"width": width, "height": height})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        for selector in ("#globalStopButton", "#startButton", "#stopButton", "[data-view]"):
            for element in page.locator(selector).all():
                box = element.bounding_box()
                assert box and box["x"] >= 0 and box["x"] + box["width"] <= width + 1
                assert box["width"] >= 44 and box["height"] >= 44, (selector, box)
        if width == 1920:
            box = page.locator("#view-trading").bounding_box()
            assert box["width"] <= 1440
            assert abs(box["x"] - (width - box["x"] - box["width"])) <= 1
        for name in ("portfolio", "activity", "system", "trading"):
            show_view(page, name)
            playwright.expect(page.locator("#globalStopButton")).to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator("#controlsPanelToggle").focus()
        page.keyboard.press("Enter")
        assert page.locator("#controlsPanel").get_attribute("open") is not None
        page.locator("#controlsPanelToggle").focus()
        page.keyboard.press("Space")
        assert page.locator("#controlsPanel").get_attribute("open") is None
        page.locator('[data-trading-tab="position"]').click()
        page.wait_for_function("""() => {
            const canvas = document.getElementById('positionChart');
            const chart = Chart.getChart(canvas);
            return chart && chart.width > 0 && Math.abs(chart.width - canvas.clientWidth) <= 1;
        }""")
        for canvas in page.locator("#view-trading canvas:visible").all():
            box = canvas.bounding_box()
            assert box["width"] > 0 and box["height"] > 0
            assert box["x"] >= 0 and box["x"] + box["width"] <= width + 1
        show_view(page, "portfolio")
        page.wait_for_function("""() => {
            const canvases = [...document.querySelectorAll('#view-portfolio canvas')]
                .filter(canvas => canvas.clientWidth > 0 && canvas.clientHeight > 0);
            return canvases.length >= 2 && canvases.every(canvas => {
                const chart = Chart.getChart(canvas);
                return chart && chart.width > 0 && Math.abs(chart.width - canvas.clientWidth) <= 1;
            });
        }""")


def test_trading_first_screen_prioritizes_live_operations(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        page.set_viewport_size({"width": 1280, "height": 800})
        for selector in ("#summaryEquity", "#summaryPnl", "#summaryOrders", "#summaryFeeds",
                         "#selectedInstrumentTitle", "#selectedReadiness", "#midPrice",
                         "#startButton", "#stopButton", "#globalStopButton", "#orderBookBody",
                         "#activeOrderCount"):
            box = page.locator(selector).bounding_box()
            assert box and box["y"] >= 0 and box["y"] + box["height"] <= 800, selector
        assert page.locator("#controlsPanel").get_attribute("open") is None
        assert page.locator("#strategyVariables").get_attribute("open") is None


def test_workbench_reduced_motion_keyboard_zoom_and_mobile_table_access(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        page.emulate_media(reduced_motion="reduce")
        page.set_viewport_size({"width": 375, "height": 812})
        page.reload()
        page.wait_for_function("isConnected && state.crossAssetMode")
        playwright.expect(page.locator("h1")).to_be_visible()
        assert page.evaluate("""() => [...document.querySelectorAll('body *')].every(element => {
            const style = getComputedStyle(element);
            return style.animationDuration.split(',').every(value => parseFloat(value) <= 0.00001)
                && style.transitionDuration.split(',').every(value => parseFloat(value) <= 0.00001);
        })""")
        page.locator('nav [data-view="portfolio"]').focus()
        page.keyboard.press("Enter")
        playwright.expect(page.locator("#view-portfolio")).to_be_visible()
        outline = page.locator('nav [data-view="portfolio"]').evaluate("element => getComputedStyle(element).outlineStyle")
        assert outline != "none"
        page.evaluate("document.documentElement.style.fontSize = '200%'")
        for name in ("portfolio", "activity", "system", "trading"):
            show_view(page, name)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            stop = page.locator("#globalStopButton").bounding_box()
            assert stop and 0 <= stop["x"] and stop["x"] + stop["width"] <= 376
            label_geometry = page.evaluate("""() => {
                const controls = [...document.querySelectorAll('.workbench-header button, .view-nav a')];
                const labels = controls.map(element => {
                    const box = element.getBoundingClientRect();
                    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
                    const rectangles = [];
                    while (walker.nextNode()) {
                        if (!walker.currentNode.textContent.trim()) continue;
                        const range = document.createRange();
                        range.selectNodeContents(walker.currentNode);
                        rectangles.push(...[...range.getClientRects()].map(rect => ({
                            left:rect.left, right:rect.right, top:rect.top, bottom:rect.bottom})));
                    }
                    return {text:element.textContent.trim(), nav:element.matches('.view-nav a'),
                        box:{left:box.left, right:box.right, top:box.top, bottom:box.bottom}, rectangles};
                });
                return {labels, headerHeight:document.querySelector('.workbench-header').getBoundingClientRect().height};
            }""")
            assert label_geometry["headerHeight"] <= 812 * 0.4, label_geometry
            for label in label_geometry["labels"]:
                for rect in label["rectangles"]:
                    box = label["box"]
                    assert rect["left"] >= box["left"] - 1 and rect["right"] <= box["right"] + 1, label
                    assert rect["top"] >= box["top"] - 1 and rect["bottom"] <= box["bottom"] + 1, label
            nav_labels = [label for label in label_geometry["labels"] if label["nav"]]
            for index, label in enumerate(nav_labels):
                for other in nav_labels[index + 1:]:
                    for first in label["rectangles"]:
                        for second in other["rectangles"]:
                            intersection_width = min(first["right"], second["right"]) - max(first["left"], second["left"])
                            intersection_height = min(first["bottom"], second["bottom"]) - max(first["top"], second["top"])
                            assert intersection_width <= 0 or intersection_height <= 0, (label, other)
        page.evaluate("document.documentElement.style.fontSize = ''")
        select_market(page, "BINANCE:BTCUSDT")
        overflow = [region for region in page.locator("#view-trading .scrollable-content:visible").all()
                    if region.evaluate("element => element.scrollWidth > element.clientWidth")]
        assert overflow
        for region in overflow:
            playwright.expect(region.locator("xpath=preceding-sibling::p[1]")).to_be_visible()
            region.focus()
            region.evaluate("element => element.scrollLeft = element.scrollWidth")
            assert region.evaluate("element => element.scrollLeft > 0")
        page.locator('[data-trading-tab="fills"]').click()
        playwright.expect(page.locator("#fillHistoryBody")).to_be_visible()


def test_variables_are_scoped_drafts_until_apply(tmp_path):
    with browser_fixture(tmp_path) as (backend, page):
        select_market(page, "NASDAQ:AAPL")
        initial = backend.service.get_dashboard_state()
        original = next(row for row in initial["instruments"] if row["instrument_id"] == "NASDAQ:AAPL")["spread_bps"]
        page.locator("#controlsPanelToggle").click()
        page.locator("#strategyVariables summary").click()
        page.evaluate("""() => {
            window.capturedCommands = [];
            const send = ws.send.bind(ws);
            ws.send = value => { capturedCommands.push(JSON.parse(value)); send(value); };
        }""")
        page.locator("#spreadBpsInput").fill("13")
        page.locator("#spreadBpsInput").press("Tab")
        playwright.expect(page.locator("#spreadBpsSlider")).to_have_value("13")
        page.locator("#inventorySkewSlider").evaluate("element => { element.value = '11'; element.dispatchEvent(new Event('input', {bubbles:true})); element.dispatchEvent(new Event('change', {bubbles:true})); }")
        playwright.expect(page.locator("#inventorySkewInput")).to_have_value("11")
        page.wait_for_timeout(400)
        assert page.evaluate("capturedCommands.filter(row => row.data?.command === 'update_params').length") == 0
        unchanged = backend.service.get_dashboard_state()
        assert next(row for row in unchanged["instruments"] if row["instrument_id"] == "NASDAQ:AAPL")["spread_bps"] == original
        page.locator("#applyVariablesButton").click()
        page.wait_for_function("capturedCommands.some(row => row.data?.command === 'update_params')")
        request = page.evaluate("capturedCommands.find(row => row.data?.command === 'update_params').data")
        assert request["payload"]["instrument_id"] == "NASDAQ:AAPL"
        assert request["payload"]["spread_bps"] == 13
        assert request["payload"]["inventory_skew_bps"] == 11
        playwright.expect(page.locator("#controlMessage")).to_have_attribute("data-outcome", "accepted")
        page.wait_for_function("crossAssetState.instruments.find(row => row.instrument_id === 'NASDAQ:AAPL').spread_bps === 13")
        btc = next(row for row in backend.service.get_dashboard_state()["instruments"] if row["instrument_id"] == "BINANCE:BTCUSDT")
        assert btc["spread_bps"] != 13


def test_command_feedback_survives_collapsed_controls_and_lost_acknowledgment(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        assert page.locator("#controlsPanel").get_attribute("open") is None
        freeze_fixture_updates(page)
        page.evaluate("""() => {
            window.capturedCommands = [];
            const send = WebSocket.prototype.send;
            WebSocket.prototype.send = function(value) {
                const message = JSON.parse(value);
                if (message?.type === 'paper_command') {
                    capturedCommands.push(message);
                    return;
                }
                return send.call(this, value);
            };
        }""")
        page.locator("#globalStopButton").click()
        playwright.expect(page.locator("#controlMessage")).to_have_attribute("data-outcome", "pending")
        page.evaluate("""() => {
            const request = capturedCommands.at(-1).data;
            pendingCommands.get(request.request_id).startedAt = Date.now() - 10001;
            checkCommandTimeouts();
        }""")
        playwright.expect(page.locator("#controlMessage")).to_have_attribute("data-outcome", "unknown")
        playwright.expect(page.locator("#controlMessage")).to_contain_text("Outcome unknown")
        playwright.expect(page.locator("#globalStopButton")).to_be_enabled()
        page.evaluate("""() => {
            const request = capturedCommands.at(-1).data;
            handleMessage({type:'control_response', action:request.command, request_id:request.request_id,
                success:false, code:'revision_conflict', message:'Fixture account changed; review latest state.'});
        }""")
        playwright.expect(page.locator("#controlMessage")).to_have_attribute("data-outcome", "rejected")
        playwright.expect(page.locator("#controlMessage")).to_contain_text("Fixture account changed")
        playwright.expect(page.locator("#globalStopButton")).to_be_enabled()
        page.locator("#globalStopButton").click()
        playwright.expect(page.locator("#controlMessage")).to_have_attribute("data-outcome", "pending")
        page.evaluate("ws.close()")
        page.wait_for_function("!isConnected")
        playwright.expect(page.locator("#controlMessage")).to_have_attribute("data-outcome", "unknown")
        playwright.expect(page.locator("#controlMessage")).not_to_contain_text("completed")
        playwright.expect(page.locator("#selectedReadiness")).to_contain_text("last received")
        playwright.expect(page.locator("#summaryFeeds")).to_contain_text("Unknown")
        playwright.expect(page.locator("#globalStopButton")).to_be_disabled()
        show_view(page, "activity")
        playwright.expect(page.locator("#commandLogBody")).to_contain_text("Fixture account changed")
        page.wait_for_function("isConnected", timeout=10000)
        page.wait_for_timeout(300)
        assert page.evaluate("capturedCommands.filter(row => row?.data?.command === 'stop_strategy').length") == 2


def test_selected_order_fill_scope_and_portfolio_totals_are_explicit(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        freeze_fixture_updates(page)
        page.evaluate("""() => {
            const data = structuredClone(crossAssetState);
            data.instruments.forEach(row => row.active_orders = []);
            data.instruments[0].active_orders = [{order_id:'ui-btc-order', instrument_id:data.instruments[0].instrument_id,
                side:'BUY', price:79000, quantity:0.001, remaining_quantity:0.001, accepted_at:Date.now()/1000}];
            data.instruments[1].active_orders = [{order_id:'ui-equity-order', instrument_id:data.instruments[1].instrument_id,
                side:'BUY', price:190, quantity:1, remaining_quantity:1, accepted_at:Date.now()/1000}];
            data.recent_fills = [
                {fill_id:'ui-btc-fill', instrument_id:data.instruments[0].instrument_id, side:'BUY', price:79000,
                    quantity:0.001, timestamp:Date.now()/1000, execution_model:'Synthetic UI fixture'},
                {fill_id:'ui-equity-fill', instrument_id:data.instruments[1].instrument_id, side:'BUY', price:190,
                    quantity:1, timestamp:Date.now()/1000, execution_model:'Synthetic UI fixture'}];
            renderCrossAssetState(data);
        }""")
        equity = page.locator("#summaryEquity").inner_text()
        pnl = page.locator("#summaryPnl").inner_text()
        select_market(page, "NASDAQ:AAPL")
        playwright.expect(page.locator("#ordersScope")).to_have_value("selected")
        playwright.expect(page.locator("#activeOrdersBody")).to_contain_text("AAPL")
        playwright.expect(page.locator("#activeOrdersBody")).not_to_contain_text("BTC/USDT")
        page.locator("#ordersScope").select_option("all")
        playwright.expect(page.locator("#activeOrdersBody")).to_contain_text("BTC/USDT")
        page.locator('[data-trading-tab="fills"]').click()
        playwright.expect(page.locator("#fillsScope")).to_have_value("selected")
        playwright.expect(page.locator("#fillHistoryBody")).to_contain_text("AAPL")
        playwright.expect(page.locator("#fillHistoryBody")).not_to_contain_text("BTC/USDT")
        page.locator("#fillsScope").select_option("all")
        playwright.expect(page.locator("#fillHistoryBody")).to_contain_text("BTC/USDT")
        select_market(page, "OANDA:EUR_USD")
        assert page.locator("#summaryEquity").inner_text() == equity
        assert page.locator("#summaryPnl").inner_text() == pnl
        playwright.expect(page.locator("#selectedInstrumentTitle")).to_contain_text("EUR/USD")


def test_hundred_instrument_watchlist_preserves_search_selection_focus_and_scroll(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        freeze_fixture_updates(page)
        page.evaluate("""() => {
            window.scaleFixture = structuredClone(crossAssetState);
            const templates = scaleFixture.instruments;
            scaleFixture.instruments = Array.from({length:100}, (_, index) => ({
                ...structuredClone(templates[index % templates.length]),
                instrument_id:`FIXTURE:${String(index).padStart(3, '0')}`,
                symbol:`TEST${String(index).padStart(3, '0')}`, active_orders:[], public_trades:[]}));
            renderCrossAssetState(scaleFixture);
        }""")
        playwright.expect(page.locator("#instrumentGrid .instrument-card")).to_have_count(100)
        page.locator("#instrumentSearch").fill("TEST")
        page.locator("#assetFilter").select_option("equity")
        playwright.expect(page.locator("#instrumentGrid .instrument-card:visible")).to_have_count(25)
        select_market(page, "FIXTURE:049")
        page.locator("#instrumentSearch").focus()
        page.locator("#instrumentGrid").evaluate("element => element.scrollTop = 220")
        scroll = page.locator("#instrumentGrid").evaluate("element => element.scrollTop")
        assert scroll > 0
        timings = page.evaluate("""() => {
            const timings = [];
            for (let tick = 0; tick < 20; tick += 1) {
                scaleFixture.instruments.forEach(row => row.mid_price += 0.01);
                const start = performance.now();
                renderCrossAssetState(scaleFixture);
                timings.push(performance.now() - start);
            }
            return timings;
        }""")
        assert max(timings) < 1000, timings
        playwright.expect(page.locator("#instrumentSearch")).to_have_value("TEST")
        playwright.expect(page.locator("#instrumentSearch")).to_be_focused()
        playwright.expect(page.locator("#assetFilter")).to_have_value("equity")
        assert page.evaluate("selectedInstrumentId") == "FIXTURE:049"
        assert abs(page.locator("#instrumentGrid").evaluate("element => element.scrollTop") - scroll) <= 1
        playwright.expect(page.locator("#instrumentGrid .instrument-card:visible")).to_have_count(25)
        frame_latency = page.evaluate("""async () => {
            const selected = scaleFixture.instruments.find(row => row.instrument_id === selectedInstrumentId);
            selected.mid_price = 987.65;
            selected.best_bid = 987.64;
            selected.best_ask = 987.66;
            selected.bids[0].price = 987.64;
            selected.asks[0].price = 987.66;
            const start = performance.now();
            renderCrossAssetState(scaleFixture);
            await new Promise(requestAnimationFrame);
            await new Promise(requestAnimationFrame);
            return performance.now() - start;
        }""")
        assert frame_latency < 1000
        playwright.expect(page.locator("#midPrice")).to_contain_text("987.65")
        output = Path(os.environ.get("PAPER_BROWSER_EVIDENCE", str(tmp_path)))
        output.mkdir(parents=True, exist_ok=True)
        (output / "watchlist_100_ui_metrics.json").write_text(json.dumps({
            "kind": "synthetic_ui_only", "instruments": 100, "updates": len(timings),
            "max_synchronous_render_ms": max(timings), "update_through_paint_ms": frame_latency,
            "live_feed_certification": False}, indent=2), encoding="utf-8")


def test_invalid_valuation_and_missing_analytics_are_unavailable_not_zero(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        freeze_fixture_updates(page)
        page.evaluate("""() => {
            const data = structuredClone(crossAssetState);
            data.portfolio = null;
            data.portfolio_error = 'Fixture conversion mark is stale';
            data.analytics = {};
            renderCrossAssetState(data);
        }""")
        playwright.expect(page.locator("#summaryEquity")).to_contain_text("Unavailable")
        playwright.expect(page.locator("#summaryPnl")).to_contain_text("Unavailable")
        show_view(page, "portfolio")
        playwright.expect(page.locator("#slippageMetricLabel")).to_have_text("5-second fill markout")
        playwright.expect(page.locator("#avgSlippage")).to_contain_text("Unavailable")
        playwright.expect(page.locator("#sharpeRatio")).to_contain_text("Unavailable")
        playwright.expect(page.locator("#portfolioBreakdown")).to_contain_text("Fixture conversion mark is stale")


def test_readiness_reasons_are_not_replaced_by_generic_waiting(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        freeze_fixture_updates(page)
        cases = {
            "stale_quote": "Quote feed is stale.",
            "market_closed": "Market session is closed.",
            "book_unsynchronized": "Order book requires resynchronization.",
            "accounting_blocked": "A sourced settlement value is required.",
            "quoting_disabled": "Quoting is disabled for this instrument.",
        }
        for code, message in cases.items():
            page.evaluate("""({code, message}) => {
                const data = structuredClone(crossAssetState);
                const selected = data.instruments.find(row => row.instrument_id === selectedInstrumentId);
                selected.market_data_ready = false;
                selected.readiness = {code, message};
                renderCrossAssetState(data);
            }""", {"code": code, "message": message})
            playwright.expect(page.locator("#selectedReadiness")).to_contain_text(message)


def test_legacy_btc_snapshot_preserves_metric_meaning_and_local_controls(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        freeze_fixture_updates(page)
        page.evaluate("""() => {
            handleFullState({type:'full_state', data:{is_running:false, market_data_ready:true,
                book_ready:true, trade_ready:true, starting_cash:10000,
                strategy_params:{risk_aversion:0.05, order_size:0.01}, strategy_interval:5},
                trading:{position:0.01, cash:9200, equity:10000, total_pnl:0, total_fees:0},
                market:{mid_price:80000, spread:0.02,
                    bids:[{price:79999.99, size:1}], asks:[{price:80000.01, size:1}]},
                websocket:{orderbook_updates:1, trade_updates:1}});
            handlePerformanceMetrics({avg_slippage_bps:1.25, fill_rate:50,
                sharpe_ratio:0.2, max_drawdown:4});
        }""")
        assert not page.evaluate("state.crossAssetMode")
        playwright.expect(page.locator("#startButton")).to_be_visible()
        page.locator("#controlsPanelToggle").click()
        page.locator("#strategyVariables summary").click()
        playwright.expect(page.locator("#legacyStrategyVariables")).to_be_visible()
        playwright.expect(page.locator("#crossAssetVariables")).not_to_be_visible()
        show_view(page, "portfolio")
        assert "slippage" in page.locator("#slippageMetricLabel").inner_text().lower()
        playwright.expect(page.locator("#avgSlippage")).to_contain_text("1.25")
        playwright.expect(page.locator("#fillRate")).to_have_text("50.0%")


def test_cross_asset_feed_status_and_retained_public_print_count_survive_timer(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        freeze_fixture_updates(page)
        page.evaluate("""() => {
            const data = structuredClone(crossAssetState);
            const btc = data.instruments.find(row => row.instrument_id === 'BINANCE:BTCUSDT');
            btc.quote_age_seconds = 0.1;
            btc.trade_age_seconds = 0.2;
            btc.market_data_ready = true;
            btc.readiness = {code:'ready', message:'Feed ready'};
            btc.public_trades = Array.from({length:3}, (_, index) => ({
                timestamp:Date.now()/1000 - index, price:80000 + index,
                quantity:0.001, side:'BUY', source_event_id:`retained-ui-print-${index}`}));
            renderCrossAssetState(data);
        }""")
        playwright.expect(page.locator("#bookUpdateStatus")).to_contain_text("Feed ready / quote")
        playwright.expect(page.locator("#marketPrintCount")).to_have_text("3")
        assert page.locator("#marketTapeBody tr").count() == 3
        playwright.expect(page.locator("#streamStatus")).to_contain_text("Quote:")
        playwright.expect(page.locator("#streamStatus")).to_contain_text("Public trades:")
        page.wait_for_timeout(1200)
        playwright.expect(page.locator("#bookUpdateStatus")).to_contain_text("Feed ready / quote")
        playwright.expect(page.locator("#marketPrintCount")).to_have_text("3")
        playwright.expect(page.locator("#streamStatus")).not_to_contain_text("waiting")
        playwright.expect(page.locator("#streamStatus")).to_contain_text("Public trades:")
        select_market(page, "OANDA:EUR_USD")
        playwright.expect(page.locator("#streamStatus")).to_contain_text("Public trades: not supplied by this feed")
        playwright.expect(page.locator("#marketPrintCount")).to_have_text("0")
        playwright.expect(page.locator("#readinessDetails")).to_contain_text("estimate")
        page.wait_for_timeout(1200)
        playwright.expect(page.locator("#streamStatus")).to_contain_text("Public trades: not supplied by this feed")


def test_mobile_orders_and_fills_are_reachable_in_one_action(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        page.set_viewport_size({"width": 375, "height": 812})
        for quick_button, table in (("#ordersQuickButton", "#activeOrdersBody"),
                                    ("#fillsQuickButton", "#fillHistoryBody")):
            page.evaluate("window.scrollTo(0, 0)")
            quick_box = page.locator(quick_button).bounding_box()
            assert quick_box and 0 <= quick_box["y"] and quick_box["y"] + quick_box["height"] <= 812
            page.locator(quick_button).click()
            playwright.expect(page.locator(table)).to_be_visible()
            page.wait_for_function("""selector => {
                const header = document.querySelector('.workbench-header').getBoundingClientRect();
                const row = document.querySelector(selector + ' tr').getBoundingClientRect();
                return row.top >= header.bottom - 1 && row.bottom <= innerHeight;
            }""", arg=table)
            header_box = page.locator(".workbench-header").bounding_box()
            table_box = page.locator(table + " tr").first.bounding_box()
            assert table_box["y"] >= header_box["y"] + header_box["height"] - 1
            assert table_box["y"] + table_box["height"] <= 812


def test_old_parameter_acknowledgment_cannot_replace_newer_account_snapshot(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        freeze_fixture_updates(page)
        select_market(page, "NASDAQ:AAPL")
        page.locator("#controlsPanelToggle").click()
        page.locator("#strategyVariables summary").click()
        page.evaluate("""() => {
            const data = structuredClone(crossAssetState);
            applicationRevision = 20;
            data.application.revision = 20;
            data.instruments.find(row => row.instrument_id === 'NASDAQ:AAPL').spread_bps = 14;
            renderCrossAssetState(data);
        }""")
        playwright.expect(page.locator("#spreadBpsInput")).to_have_value("14")
        page.locator("#spreadBpsInput").fill("13")
        playwright.expect(page.locator("#spreadBpsInput")).to_have_value("13")
        page.evaluate("""() => {
            trackCommand('old-parameter-ack', 'update_params',
                {instrument_id:'NASDAQ:AAPL', spread_bps:13}, false);
            handleMessage({type:'control_response', action:'update_params', request_id:'old-parameter-ack',
                success:true, revision:19, data:{updated:{instrument_id:'NASDAQ:AAPL', spread_bps:13}}});
        }""")
        assert page.evaluate("applicationRevision") == 20
        assert page.evaluate("crossAssetState.instruments.find(row => row.instrument_id === 'NASDAQ:AAPL').spread_bps") == 14
        assert not page.evaluate("variableDrafts.has('NASDAQ:AAPL')")
        playwright.expect(page.locator("#spreadBpsInput")).to_have_value("14")
        playwright.expect(page.locator("#spreadBpsSlider")).to_have_value("14")


def test_idle_disconnect_replaces_connected_feedback_and_marks_last_received_values(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        page.wait_for_function("pendingCommands.size === 0")
        playwright.expect(page.locator("#controlMessage")).to_have_attribute("data-outcome", "connected")
        equity = page.locator("#summaryEquity").inner_text()
        freeze_fixture_updates(page)
        page.evaluate("ws.close()")
        page.wait_for_function("!isConnected")
        playwright.expect(page.locator("#statusText")).to_contain_text("disconnected")
        playwright.expect(page.locator("#controlMessage")).to_have_attribute("data-outcome", "disconnected")
        playwright.expect(page.locator("#controlMessage")).to_contain_text("last received")
        playwright.expect(page.locator("#controlMessage")).not_to_have_text("Connected to the local paper backend.")
        playwright.expect(page.locator("#summaryEquity")).to_have_text(equity)
        playwright.expect(page.locator("#selectedReadiness")).to_contain_text("last received")
        playwright.expect(page.locator("#strategyStatusText")).to_contain_text("Last known:")
        playwright.expect(page.locator("#globalStopButton")).to_be_disabled()
        page.wait_for_function("isConnected && !waitingForFreshState", timeout=10000)
        playwright.expect(page.locator("#statusText")).to_have_text("Connected to local paper backend")
        playwright.expect(page.locator("#controlMessage")).to_have_attribute("data-outcome", "connected")
        playwright.expect(page.locator("#selectedReadiness")).not_to_contain_text("last received")
        playwright.expect(page.locator("#summaryFeeds")).to_have_text("4 / 4")
        playwright.expect(page.locator("#globalStopButton")).to_be_enabled()



