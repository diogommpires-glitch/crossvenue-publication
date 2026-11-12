"""Test fractional quantity preservation in Binance connector.

Verifies that fractional quantities (e.g., 0.5 BTC) are preserved through
the order book parsing pipeline without being truncated to integers.
"""

import json
import pytest
from unittest.mock import Mock, patch
from src.connectors.binance_connector import BinanceConnector
from src.utils.types import Quantity, Price


class TestFractionalQuantities:
    """Test that fractional quantities are preserved correctly."""

    def test_fractional_bid_quantity_preserved(self):
        """Verify that fractional bid quantities like 0.5 BTC are not truncated."""
        connector = BinanceConnector("BTCUSDT")

        # Mock WebSocket message with fractional quantity
        mock_message = json.dumps({
            "lastUpdateId": 123456,
            "bids": [
                ["50000.00", "0.5"],  # 0.5 BTC at $50,000
                ["49999.00", "1.25"],
                ["49998.00", "0.75"]
            ],
            "asks": [
                ["50001.00", "2.0"],
                ["50002.00", "1.5"]
            ]
        })

        # Process the message
        connector._on_ws_message(None, mock_message)

        # Get the order book
        orderbook = connector._current_orderbook
        assert orderbook is not None

        # Verify fractional quantities are preserved
        assert len(orderbook.bids) == 3
        assert float(orderbook.bids[0][1]) == 0.5, "0.5 BTC should be preserved, not truncated to 0"
        assert float(orderbook.bids[1][1]) == 1.25
        assert float(orderbook.bids[2][1]) == 0.75

    def test_fractional_ask_quantity_preserved(self):
        """Verify that fractional ask quantities are not truncated."""
        connector = BinanceConnector("BTCUSDT")

        # Mock WebSocket message with fractional quantities in asks
        mock_message = json.dumps({
            "lastUpdateId": 123457,
            "bids": [
                ["50000.00", "1.0"]
            ],
            "asks": [
                ["50001.00", "0.3"],  # 0.3 BTC
                ["50002.00", "0.75"],
                ["50003.00", "0.123456"]
            ]
        })

        connector._on_ws_message(None, mock_message)
        orderbook = connector._current_orderbook

        assert orderbook is not None
        assert len(orderbook.asks) == 3
        assert float(orderbook.asks[0][1]) == 0.3, "0.3 BTC should be preserved"
        assert float(orderbook.asks[1][1]) == 0.75
        assert abs(float(orderbook.asks[2][1]) - 0.123456) < 1e-6

    def test_quantity_type_is_float_not_int(self):
        """Verify that quantities are float-based (Quantity NewType), not int."""
        connector = BinanceConnector("BTCUSDT")

        mock_message = json.dumps({
            "lastUpdateId": 123458,
            "bids": [["50000.00", "0.5"]],
            "asks": [["50001.00", "0.3"]]
        })

        connector._on_ws_message(None, mock_message)
        orderbook = connector._current_orderbook

        # Verify type is float (Quantity is NewType of float), not int
        bid_qty = orderbook.bids[0][1]
        ask_qty = orderbook.asks[0][1]

        # Quantity is a NewType of float, so underlying type should be float
        assert isinstance(float(bid_qty), float)
        assert isinstance(float(ask_qty), float)
        assert not isinstance(bid_qty, int)
        assert not isinstance(ask_qty, int)

        # Verify they maintain fractional values
        assert float(bid_qty) == 0.5
        assert float(ask_qty) == 0.3

    def test_zero_point_five_btc_specific_case(self):
        """Specific test case for 0.5 BTC as mentioned in requirements."""
        connector = BinanceConnector("BTCUSDT")

        # Test with exactly 0.5 BTC
        mock_message = json.dumps({
            "lastUpdateId": 999999,
            "bids": [["50000.00", "0.5"]],
            "asks": [["50001.00", "0.5"]]
        })

        connector._on_ws_message(None, mock_message)
        orderbook = connector._current_orderbook

        # Verify 0.5 is preserved exactly
        bid_qty = float(orderbook.bids[0][1])
        ask_qty = float(orderbook.asks[0][1])

        assert bid_qty == 0.5, f"Expected 0.5, got {bid_qty}"
        assert ask_qty == 0.5, f"Expected 0.5, got {ask_qty}"

        # If int() was mistakenly used, these would be 0
        assert bid_qty != 0, "Quantity should not be truncated to 0"
        assert ask_qty != 0, "Quantity should not be truncated to 0"
