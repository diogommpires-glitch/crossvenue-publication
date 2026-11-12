"""Authentication does not change the legacy account's command protocol."""

import pytest

from tests.test_platform_browser import browser_fixture, freeze_fixture_updates

pytestmark = pytest.mark.browser


def test_authenticated_legacy_dashboard_sends_legacy_commands(tmp_path):
    with browser_fixture(tmp_path) as (_, page):
        freeze_fixture_updates(page)
        messages = page.evaluate("""() => {
            handleFullState({data: {is_running: false, market_data_ready: true},
                trading: {equity: 10000, cash: 10000, position: 0, total_pnl: 0}});
            const messages = [];
            ws = {readyState: WebSocket.OPEN, send: raw => messages.push(JSON.parse(raw))};
            sendCommand('start_strategy');
            sendCommand('update_params', {params: {risk_aversion: 0.07}});
            sendCommand('stop_strategy');
            return messages;
        }""")
        assert messages == [
            {"type": "start_strategy"},
            {"type": "update_params", "params": {"risk_aversion": 0.07}},
            {"type": "stop_strategy"},
        ]
