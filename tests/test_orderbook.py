"""Comprehensive tests for the limit order book.

These tests verify:
- Price-time priority enforcement
- Correct fill generation
- Self-matching prevention
- Edge cases (empty book, crossing orders, partial fills)
"""

import pytest
from src.orderbook.order import Order, Fill
from src.orderbook.limit_order_book import LimitOrderBook
from src.utils.types import Side, OrderType, OrderStatus, Price, Quantity, Timestamp


class TestOrderCreation:
    """Test Order class validation and creation."""

    def test_valid_limit_order(self):
        """Test creating a valid limit order."""
        order = Order(
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(100),
            price=Price(99.50),
            timestamp=Timestamp(0.0),
            trader_id="trader1",
        )
        assert order.quantity == 100
        assert order.price == 99.50
        assert order.side == Side.BUY

    def test_invalid_zero_quantity(self):
        """Test that zero quantity raises error."""
        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                quantity=Quantity(0),
                price=Price(100.0),
                timestamp=Timestamp(0.0),
                trader_id="trader1",
            )

    def test_limit_order_without_price(self):
        """Test that limit orders require a price."""
        with pytest.raises(ValueError, match="must have a price"):
            Order(
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                quantity=Quantity(100),
                price=None,
                timestamp=Timestamp(0.0),
                trader_id="trader1",
            )

    def test_market_order_with_price(self):
        """Test that market orders cannot have a price."""
        with pytest.raises(ValueError, match="cannot have a price"):
            Order(
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=Quantity(100),
                price=Price(100.0),
                timestamp=Timestamp(0.0),
                trader_id="trader1",
            )


class TestOrderBook:
    """Test LimitOrderBook matching logic."""

    @pytest.fixture
    def empty_book(self):
        """Create an empty order book."""
        return LimitOrderBook()

    def test_empty_book_state(self, empty_book):
        """Test initial state of empty book."""
        assert empty_book.get_best_bid() is None
        assert empty_book.get_best_ask() is None
        assert empty_book.get_mid_price() is None
        assert empty_book.get_spread() is None

    def test_add_single_bid(self, empty_book):
        """Test adding a single buy order."""
        order = Order(
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(100),
            price=Price(99.0),
            timestamp=Timestamp(1.0),
            trader_id="trader1",
        )

        final_order, fills = empty_book.add_order(order)

        assert final_order.status == OrderStatus.ACTIVE
        assert len(fills) == 0
        assert empty_book.get_best_bid() == 99.0
        assert empty_book.get_best_ask() is None

    def test_add_single_ask(self, empty_book):
        """Test adding a single sell order."""
        order = Order(
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=Quantity(100),
            price=Price(101.0),
            timestamp=Timestamp(1.0),
            trader_id="trader1",
        )

        final_order, fills = empty_book.add_order(order)

        assert final_order.status == OrderStatus.ACTIVE
        assert len(fills) == 0
        assert empty_book.get_best_bid() is None
        assert empty_book.get_best_ask() == 101.0

    def test_non_crossing_orders(self, empty_book):
        """Test that non-crossing orders don't match."""
        # Add bid at 99
        bid = Order(
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(100),
            price=Price(99.0),
            timestamp=Timestamp(1.0),
            trader_id="trader1",
        )
        empty_book.add_order(bid)

        # Add ask at 101 (no cross)
        ask = Order(
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=Quantity(100),
            price=Price(101.0),
            timestamp=Timestamp(2.0),
            trader_id="trader2",
        )
        final_order, fills = empty_book.add_order(ask)

        assert final_order.status == OrderStatus.ACTIVE
        assert len(fills) == 0
        assert empty_book.get_best_bid() == 99.0
        assert empty_book.get_best_ask() == 101.0
        assert empty_book.get_spread() == 2.0

    def test_simple_match(self, empty_book):
        """Test a simple crossing trade."""
        # Add resting bid at 100
        bid = Order(
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(100),
            price=Price(100.0),
            timestamp=Timestamp(1.0),
            trader_id="trader1",
        )
        empty_book.add_order(bid)

        # Add aggressive ask at 100 (crosses)
        ask = Order(
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=Quantity(100),
            price=Price(100.0),
            timestamp=Timestamp(2.0),
            trader_id="trader2",
        )
        final_order, fills = empty_book.add_order(ask)

        # Should fully fill
        assert final_order.status == OrderStatus.FILLED
        assert len(fills) == 2  # One for each side

        # Check aggressor fill
        aggressor_fills = [f for f in fills if f.is_aggressor]
        assert len(aggressor_fills) == 1
        assert aggressor_fills[0].side == Side.SELL
        assert aggressor_fills[0].quantity == 100
        assert aggressor_fills[0].price == 100.0

        # Check passive fill
        passive_fills = [f for f in fills if not f.is_aggressor]
        assert len(passive_fills) == 1
        assert passive_fills[0].side == Side.BUY
        assert passive_fills[0].quantity == 100
        assert passive_fills[0].price == 100.0

        # Book should be empty
        assert empty_book.get_best_bid() is None
        assert empty_book.get_best_ask() is None

    def test_partial_fill(self, empty_book):
        """Test partial fill when resting order is smaller."""
        # Add resting bid for 50
        bid = Order(
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(50),
            price=Price(100.0),
            timestamp=Timestamp(1.0),
            trader_id="trader1",
        )
        empty_book.add_order(bid)

        # Add aggressive ask for 100 (larger)
        ask = Order(
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=Quantity(100),
            price=Price(100.0),
            timestamp=Timestamp(2.0),
            trader_id="trader2",
        )
        final_order, fills = empty_book.add_order(ask)

        # Ask should be partially filled
        assert final_order.status == OrderStatus.PARTIALLY_FILLED
        assert final_order.quantity == 50  # Remaining quantity
        assert len(fills) == 2

        # 50 shares traded
        assert sum(f.quantity for f in fills if f.is_aggressor) == 50

        # Remaining 50 should be in book
        assert empty_book.get_best_ask() == 100.0
        depth = empty_book.get_depth(Side.SELL, levels=1)
        assert depth[0][1] == 50  # 50 shares at best ask

    def test_time_priority(self, empty_book):
        """Test that earlier orders fill first at same price."""
        # Add two bids at same price
        bid1 = Order(
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(30),
            price=Price(100.0),
            timestamp=Timestamp(1.0),
            trader_id="trader1",
        )
        empty_book.add_order(bid1)

        bid2 = Order(
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(70),
            price=Price(100.0),
            timestamp=Timestamp(2.0),
            trader_id="trader2",
        )
        empty_book.add_order(bid2)

        # Add ask that crosses
        ask = Order(
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=Quantity(50),
            price=Price(100.0),
            timestamp=Timestamp(3.0),
            trader_id="trader3",
        )
        final_order, fills = empty_book.add_order(ask)

        # First bid should fill completely (30), second partially (20)
        assert final_order.status == OrderStatus.FILLED
        assert len(fills) == 4  # Two fills (bid1 complete, bid2 partial)

        # Check that bid1 filled completely
        bid1_fills = [f for f in fills if f.order_id == bid1.order_id]
        assert len(bid1_fills) == 1
        assert bid1_fills[0].quantity == 30

        # Check that bid2 filled partially
        bid2_fills = [f for f in fills if f.order_id == bid2.order_id]
        assert len(bid2_fills) == 1
        assert bid2_fills[0].quantity == 20

        # Remaining 50 from bid2 should still be in book
        depth = empty_book.get_depth(Side.BUY, levels=1)
        assert depth[0][1] == 50

    def test_price_priority(self, empty_book):
        """Test that better prices fill first."""
        # Add bid at 100
        bid1 = Order(
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(50),
            price=Price(100.0),
            timestamp=Timestamp(1.0),
            trader_id="trader1",
        )
        empty_book.add_order(bid1)

        # Add bid at 101 (better price, but later time)
        bid2 = Order(
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(50),
            price=Price(101.0),
            timestamp=Timestamp(2.0),
            trader_id="trader2",
        )
        empty_book.add_order(bid2)

        # Add ask at 100 (should match bid2 first)
        ask = Order(
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=Quantity(50),
            price=Price(100.0),
            timestamp=Timestamp(3.0),
            trader_id="trader3",
        )
        final_order, fills = empty_book.add_order(ask)

        # Should fill with bid2 at 101
        assert final_order.status == OrderStatus.FILLED
        passive_fills = [f for f in fills if not f.is_aggressor]
        assert len(passive_fills) == 1
        assert passive_fills[0].price == 101.0
        assert passive_fills[0].order_id == bid2.order_id

    def test_self_matching_prevention(self, empty_book):
        """Test that same trader cannot match with themselves."""
        # Add bid from trader1
        bid = Order(
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(100),
            price=Price(100.0),
            timestamp=Timestamp(1.0),
            trader_id="trader1",
        )
        empty_book.add_order(bid)

        # Try to add ask from same trader
        ask = Order(
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=Quantity(100),
            price=Price(100.0),
            timestamp=Timestamp(2.0),
            trader_id="trader1",
        )
        final_order, fills = empty_book.add_order(ask)

        # Should not match (self-match prevented)
        # Note: Current implementation skips self-orders, so ask remains active
        assert final_order.status == OrderStatus.ACTIVE
        assert len(fills) == 0

    def test_cancel_order(self, empty_book):
        """Test order cancellation."""
        # Add order
        order = Order(
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(100),
            price=Price(100.0),
            timestamp=Timestamp(1.0),
            trader_id="trader1",
        )
        final_order, _ = empty_book.add_order(order)

        # Cancel it
        assert empty_book.cancel_order(final_order.order_id) is True

        # Should be removed from book
        assert empty_book.get_best_bid() is None

        # Cannot cancel twice
        assert empty_book.cancel_order(final_order.order_id) is False

    def test_market_order_full_fill(self, empty_book):
        """Test market order that fully fills."""
        # Add resting ask
        ask = Order(
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=Quantity(100),
            price=Price(101.0),
            timestamp=Timestamp(1.0),
            trader_id="trader1",
        )
        empty_book.add_order(ask)

        # Market buy should fill
        market_buy = Order(
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(100),
            price=None,
            timestamp=Timestamp(2.0),
            trader_id="trader2",
        )
        final_order, fills = empty_book.add_order(market_buy)

        assert final_order.status == OrderStatus.FILLED
        assert len(fills) == 2
        assert fills[0].price == 101.0  # Filled at ask price

    def test_market_order_reject_insufficient_liquidity(self, empty_book):
        """Test market order rejection when insufficient liquidity (v0.3.0: verify atomic rejection)."""
        # Add small resting ask
        ask = Order(
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=Quantity(50),
            price=Price(101.0),
            timestamp=Timestamp(1.0),
            trader_id="trader1",
        )
        empty_book.add_order(ask)

        # Capture book state before rejection
        best_ask_before = empty_book.get_best_ask()
        depth_before = empty_book.get_depth(Side.SELL, levels=5)

        # Large market buy should reject
        market_buy = Order(
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(100),
            price=None,
            timestamp=Timestamp(2.0),
            trader_id="trader2",
        )
        final_order, fills = empty_book.add_order(market_buy)

        # Should reject (cannot fully fill)
        assert final_order.status == OrderStatus.REJECTED
        assert len(fills) == 0

        # v0.3.0: Book state must be unchanged after rejection (atomic guarantee)
        assert empty_book.get_best_ask() == best_ask_before
        assert empty_book.get_depth(Side.SELL, levels=5) == depth_before
        assert empty_book.total_fills == 0  # No fills should have been recorded
        assert empty_book.total_volume == 0

    def test_depth_levels(self, empty_book):
        """Test market depth retrieval."""
        # Add multiple levels
        for i, price in enumerate([98.0, 99.0, 100.0]):
            order = Order(
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                quantity=Quantity((i + 1) * 100),
                price=Price(price),
                timestamp=Timestamp(float(i)),
                trader_id=f"trader{i}",
            )
            empty_book.add_order(order)

        # Check depth
        depth = empty_book.get_depth(Side.BUY, levels=3)
        assert len(depth) == 3

        # Should be in descending price order (best first)
        assert depth[0] == (Price(100.0), Quantity(300))
        assert depth[1] == (Price(99.0), Quantity(200))
        assert depth[2] == (Price(98.0), Quantity(100))


class TestFillRecords:
    """Test Fill record creation and validation."""

    def test_valid_fill(self):
        """Test creating a valid fill."""
        fill = Fill(
            order_id="order1",
            side=Side.BUY,
            quantity=Quantity(100),
            price=Price(99.50),
            timestamp=Timestamp(1.0),
            trader_id="trader1",
            is_aggressor=True,
        )
        assert fill.quantity == 100
        assert fill.price == 99.50
        assert fill.is_aggressor is True

    def test_invalid_fill_quantity(self):
        """Test that invalid fill quantity raises error."""
        with pytest.raises(ValueError, match="quantity must be positive"):
            Fill(
                order_id="order1",
                side=Side.BUY,
                quantity=Quantity(0),
                price=Price(99.50),
                timestamp=Timestamp(1.0),
                trader_id="trader1",
                is_aggressor=True,
            )
