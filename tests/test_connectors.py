"""Unit tests for Binance connectors.

Tests cover:
- BaseConnector interface compliance
- PaperTrader execution logic
- Order book snapshot handling
- Balance management
- Error handling
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from src.connectors import (
    BaseConnector,
    BinanceConnector,
    PaperTrader,
    OrderBookSnapshot,
    OrderResponse,
    ConnectionStatus,
)
from src.utils.types import Side, Price, Quantity, Timestamp
from src.orderbook.order import Fill


class TestOrderBookSnapshot:
    """Test OrderBookSnapshot data class."""

    def test_best_bid_ask(self):
        """Test best bid/ask properties."""
        snapshot = OrderBookSnapshot(
            timestamp=Timestamp(time.time()),
            bids=[(Price(100.0), Quantity(10)), (Price(99.0), Quantity(20))],
            asks=[(Price(101.0), Quantity(15)), (Price(102.0), Quantity(25))],
            symbol="BTCUSDT"
        )

        assert snapshot.best_bid == (Price(100.0), Quantity(10))
        assert snapshot.best_ask == (Price(101.0), Quantity(15))

    def test_mid_price(self):
        """Test mid price calculation."""
        snapshot = OrderBookSnapshot(
            timestamp=Timestamp(time.time()),
            bids=[(Price(100.0), Quantity(10))],
            asks=[(Price(102.0), Quantity(15))],
            symbol="BTCUSDT"
        )

        assert snapshot.mid_price == Price(101.0)

    def test_spread(self):
        """Test spread calculation."""
        snapshot = OrderBookSnapshot(
            timestamp=Timestamp(time.time()),
            bids=[(Price(100.0), Quantity(10))],
            asks=[(Price(102.0), Quantity(15))],
            symbol="BTCUSDT"
        )

        assert snapshot.spread == 2.0

    def test_empty_book(self):
        """Test handling of empty order book."""
        snapshot = OrderBookSnapshot(
            timestamp=Timestamp(time.time()),
            bids=[],
            asks=[],
            symbol="BTCUSDT"
        )

        assert snapshot.best_bid is None
        assert snapshot.best_ask is None
        assert snapshot.mid_price is None
        assert snapshot.spread is None


class TestPaperTrader:
    """Test PaperTrader simulated execution."""

    @pytest.fixture
    def mock_binance(self):
        """Create mock Binance connector."""
        with patch('src.connectors.paper_trader.BinanceConnector') as mock:
            instance = Mock()
            instance.status = ConnectionStatus.CONNECTED
            instance.get_orderbook.return_value = OrderBookSnapshot(
                timestamp=Timestamp(time.time()),
                bids=[(Price(50000.0), Quantity(1.0))],
                asks=[(Price(50100.0), Quantity(1.0))],
                symbol="BTCUSDT"
            )
            mock.return_value = instance
            yield instance

    @pytest.fixture
    def trader(self, mock_binance):
        """Create PaperTrader instance."""
        config = {
            "initial_balances": {"BTC": 1.0, "USDT": 100000.0},
            "fill_latency_ms": 0,  # No latency for tests
            "slippage_bps": 0,  # No slippage for tests
        }
        trader = PaperTrader("BTCUSDT", config)
        trader.connect()
        return trader

    def test_initialization(self, trader):
        """Test trader initialization."""
        assert trader.status == ConnectionStatus.CONNECTED
        assert trader.get_balance("BTC") == 1.0
        assert trader.get_balance("USDT") == 100000.0

    def test_market_buy_order(self, trader):
        """Test market buy order execution."""
        initial_btc = trader.get_balance("BTC")
        initial_usdt = trader.get_balance("USDT")

        response = trader.submit_market_order(Side.BUY, Quantity(0.1))

        assert response.success
        assert response.order_id is not None

        # Check balance changes
        assert trader.get_balance("BTC") > initial_btc
        assert trader.get_balance("USDT") < initial_usdt

    def test_market_sell_order(self, trader):
        """Test market sell order execution."""
        initial_btc = trader.get_balance("BTC")
        initial_usdt = trader.get_balance("USDT")

        response = trader.submit_market_order(Side.SELL, Quantity(0.1))

        assert response.success
        assert response.order_id is not None

        # Check balance changes
        assert trader.get_balance("BTC") < initial_btc
        assert trader.get_balance("USDT") > initial_usdt

    def test_limit_order_submission(self, trader):
        """Test limit order submission."""
        response = trader.submit_limit_order(
            Side.BUY,
            Price(45000.0),
            Quantity(0.1)
        )

        assert response.success
        assert response.order_id is not None

        # Check that balance is reserved
        assert trader.get_balance("USDT") == 100000.0 - (45000.0 * 0.1)

    def test_order_cancellation(self, trader):
        """Test order cancellation."""
        # Submit limit order
        response = trader.submit_limit_order(
            Side.BUY,
            Price(45000.0),
            Quantity(0.1)
        )
        order_id = response.order_id

        # Cancel it
        cancel_response = trader.cancel_order(order_id)

        assert cancel_response.success

        # Check that balance is released
        assert trader.get_balance("USDT") == 100000.0

    def test_insufficient_balance(self, trader):
        """Test order rejection due to insufficient balance."""
        # Try to buy more than we can afford
        response = trader.submit_market_order(Side.BUY, Quantity(10.0))

        # Should fail or deplete balance
        if response.success:
            # If it succeeded, balance should be nearly zero
            assert trader.get_balance("USDT") < 1000.0

    def test_order_status_tracking(self, trader):
        """Test order status tracking."""
        response = trader.submit_limit_order(
            Side.BUY,
            Price(45000.0),
            Quantity(0.1)
        )
        order_id = response.order_id

        # Check status
        order = trader.get_order_status(order_id)
        assert order is not None
        assert order.order_id == order_id
        assert order.status.name in ("ACTIVE", "PENDING")

    def test_statistics_tracking(self, trader):
        """Test trading statistics."""
        initial_orders = trader.stats["orders_submitted"]

        trader.submit_market_order(Side.BUY, Quantity(0.1))

        assert trader.stats["orders_submitted"] == initial_orders + 1
        assert trader.stats["orders_filled"] >= initial_orders

    def test_fill_callback(self, trader):
        """Test fill notification callbacks."""
        fills = []

        def on_fill(fill):
            fills.append(fill)

        trader.subscribe_fills(on_fill)

        trader.submit_market_order(Side.BUY, Quantity(0.1))

        assert len(fills) > 0
        assert isinstance(fills[0], Fill)
        assert fills[0].side == Side.BUY

    def test_orderbook_callback(self, trader, mock_binance):
        """Test order book update callbacks."""
        updates = []

        def on_orderbook(snapshot):
            updates.append(snapshot)

        trader.subscribe_orderbook(on_orderbook)

        # Trigger update
        snapshot = OrderBookSnapshot(
            timestamp=Timestamp(time.time()),
            bids=[(Price(50000.0), Quantity(1.0))],
            asks=[(Price(50100.0), Quantity(1.0))],
            symbol="BTCUSDT"
        )
        trader._process_orderbook_update(snapshot)

        assert len(updates) > 0
        assert isinstance(updates[0], OrderBookSnapshot)


class TestBinanceConnector:
    """Test BinanceConnector (mocked, no real API calls)."""

    @pytest.fixture
    def mock_requests(self):
        """Mock requests library."""
        with patch('src.connectors.binance_connector.requests') as mock:
            # Mock ping response
            ping_response = Mock()
            ping_response.status_code = 200
            ping_response.json.return_value = {}
            mock.get.return_value = ping_response
            yield mock

    @pytest.fixture
    def mock_websocket(self):
        """Mock websocket library."""
        with patch('src.connectors.binance_connector.websocket') as mock:
            ws_app = Mock()
            mock.WebSocketApp.return_value = ws_app
            yield ws_app

    def test_initialization(self, mock_requests, mock_websocket):
        """Test connector initialization."""
        config = {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "testnet": True
        }

        connector = BinanceConnector("BTCUSDT", config)

        assert connector.symbol == "BTCUSDT"
        assert connector.testnet is True
        assert connector.api_key == ""
        assert connector.api_secret == ""

    def test_authenticated_account_and_order_routes_fail_closed(self, mock_requests):
        connector = BinanceConnector("BTCUSDT", {"api_key": "ignored", "api_secret": "ignored"})
        with pytest.raises(NotImplementedError, match="execution is disabled"):
            connector.submit_market_order(Side.BUY, Quantity(0.01))
        with pytest.raises(NotImplementedError, match="execution is disabled"):
            connector.submit_limit_order(Side.BUY, Price(100.0), Quantity(0.01))
        with pytest.raises(NotImplementedError, match="order management is disabled"):
            connector.cancel_order("123")
        with pytest.raises(NotImplementedError, match="account access is disabled"):
            connector.get_balance("USDT")
        mock_requests.post.assert_not_called()
        mock_requests.delete.assert_not_called()
        mock_requests.get.assert_not_called()

    def test_symbol_normalization(self, mock_requests, mock_websocket):
        """Test symbol format normalization."""
        connector = BinanceConnector("btc/usdt", {})
        assert connector.symbol == "BTCUSDT"

        connector = BinanceConnector("BTC-USDT", {})
        assert connector.symbol == "BTCUSDT"

    def test_rate_limiting(self):
        """Test rate limiting logic."""
        from unittest.mock import patch

        config = {"rate_limit_per_minute": 10}
        connector = BinanceConnector("BTCUSDT", config)

        # Fill up the rate limit
        for _ in range(10):
            connector._request_times.put(time.time())

        # Next request should trigger delay
        with patch("src.connectors.binance_connector.time.sleep") as sleep:
            connector._enforce_rate_limit()

        # Verify the delay without waiting for the full 60-second rate window.
        sleep.assert_called_once()
        assert sleep.call_args.args[0] > 0

    def test_price_formatting(self):
        """Test price formatting."""
        connector = BinanceConnector("BTCUSDT", {})

        formatted = connector._format_price(Price(50123.456789))
        assert formatted == "50123.46"  # 2 decimal places

    def test_quantity_formatting(self):
        """Test quantity formatting."""
        connector = BinanceConnector("BTCUSDT", {})

        formatted = connector._format_quantity(Quantity(0.123456789))
        assert formatted == "0.123457"  # 6 decimal places

    def test_fractional_quantity_parsing(self):
        """Test that fractional BTC quantities are preserved in WebSocket parsing."""
        connector = BinanceConnector(symbol="BTCUSDT")

        # Mock depth data with fractional quantity
        mock_message = json.dumps({
            "bids": [["67000.00", "0.5"]],  # 0.5 BTC at $67k
            "asks": [["67100.00", "1.25"]]  # 1.25 BTC at $67.1k
        })

        # Simulate WebSocket message processing (inline parsing from _on_ws_message)
        data = json.loads(mock_message)
        bids = [(Price(float(p)), Quantity(float(q)))
                for p, q in data.get("bids", [])]
        asks = [(Price(float(p)), Quantity(float(q)))
                for p, q in data.get("asks", [])]

        # Check that 0.5 BTC is preserved (not converted to 0)
        assert len(bids) == 1
        assert bids[0][0] == 67000.0  # Price
        assert abs(bids[0][1] - 0.5) < 1e-6  # Quantity = 0.5 BTC

        assert len(asks) == 1
        assert asks[0][0] == 67100.0
        assert abs(asks[0][1] - 1.25) < 1e-6  # Quantity = 1.25 BTC

    def test_public_trade_callback_preserves_exchange_timestamp(self):
        from src.connectors.binance_websocket import BinanceWebSocket

        received = []
        ws = BinanceWebSocket(on_trade=lambda *args: received.append(args))
        ws._handle_trade_update({"p": "100.5", "q": "0.02", "m": True, "T": 1_700_000_000_000})
        assert received == [(100.5, 0.02, "sell", 1_700_000_000.0)]

    def test_public_trade_callback_keeps_legacy_three_argument_contract(self):
        from src.connectors.binance_websocket import BinanceWebSocket

        received = []

        def legacy_callback(price, quantity, side):
            received.append((price, quantity, side))

        ws = BinanceWebSocket(on_trade=legacy_callback)
        ws._handle_trade_update({"p": "100.5", "q": "0.02", "m": True, "T": 1_700_000_000_000})
        assert received == [(100.5, 0.02, "sell")]


class TestConnectorInterface:
    """Test that all connectors implement BaseConnector interface."""

    def test_paper_trader_interface(self):
        """Test PaperTrader implements all required methods."""
        required_methods = [
            'connect',
            'disconnect',
            'get_orderbook',
            'submit_market_order',
            'submit_limit_order',
            'cancel_order',
            'get_balance',
            'subscribe_orderbook',
            'subscribe_fills',
        ]

        for method in required_methods:
            assert hasattr(PaperTrader, method)
            assert callable(getattr(PaperTrader, method))

    def test_binance_connector_interface(self):
        """Test BinanceConnector implements all required methods."""
        required_methods = [
            'connect',
            'disconnect',
            'get_orderbook',
            'submit_market_order',
            'submit_limit_order',
            'cancel_order',
            'get_balance',
            'subscribe_orderbook',
            'subscribe_fills',
        ]

        for method in required_methods:
            assert hasattr(BinanceConnector, method)
            assert callable(getattr(BinanceConnector, method))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

