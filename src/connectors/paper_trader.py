"""Paper trading connector that simulates execution against real Binance prices.

This connector provides:
- Real market data from Binance (via BinanceConnector)
- Local simulated execution with configurable fill logic
- Position and balance tracking
- No actual exchange interaction for orders

Use it for paper strategy development and evaluation against public market data.
"""

import time
import uuid
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .base_connector import (
    BaseConnector,
    ConnectionStatus,
    OrderBookSnapshot,
    OrderResponse,
)
from .binance_connector import BinanceConnector
from ..utils.types import Side, OrderType, Price, Quantity, Timestamp
from ..orderbook.order import Fill, Order, OrderStatus


@dataclass
class SimulatedOrder:
    """Internal representation of a simulated order."""
    order_id: str
    side: Side
    order_type: OrderType
    quantity: Quantity
    price: Optional[Price]
    timestamp: Timestamp
    client_order_id: Optional[str]
    status: OrderStatus = OrderStatus.ACTIVE
    filled_quantity: Quantity = Quantity(0)

    @property
    def remaining_quantity(self) -> Quantity:
        """Unfilled quantity."""
        return Quantity(self.quantity - self.filled_quantity)

    @property
    def is_complete(self) -> bool:
        """Whether order is completely filled."""
        return self.filled_quantity >= self.quantity


class PaperTrader(BaseConnector):
    """Paper trading connector with simulated execution.

    This connector:
    - Uses real Binance market data
    - Simulates order fills based on current prices
    - Tracks virtual balances and positions
    - Never sends orders to the real exchange

    Config keys:
        binance_config: Config dict to pass to BinanceConnector
        initial_balances: Dict of {asset: quantity} starting balances
        fill_latency_ms: Simulated order execution delay (default: 50ms)
        slippage_bps: Simulated slippage in basis points (default: 5)
        enable_partial_fills: Allow partial fills for limit orders (default: True)
    """

    def __init__(self, symbol: str, config: Optional[Dict] = None):
        """Initialize paper trading connector.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            config: Configuration dict
        """
        super().__init__(symbol, config)

        # Normalize None config to empty dict
        if config is None:
            config = {}

        # Market data connector (real Binance data)
        binance_config = config.get("binance_config", {})
        self._binance = BinanceConnector(symbol, binance_config)

        # Simulation parameters
        self.fill_latency_ms = config.get("fill_latency_ms", 50)
        self.slippage_bps = config.get("slippage_bps", 5)  # 5 bps = 0.05%
        self.enable_partial_fills = config.get("enable_partial_fills", True)

        # Virtual balances
        initial_balances = config.get("initial_balances", {})
        self._balances: Dict[str, float] = defaultdict(float, initial_balances)

        # Extract base and quote from symbol (e.g., BTCUSDT -> BTC, USDT)
        self.base_asset = symbol[:3]  # Simplified: assumes 3-letter base
        self.quote_asset = symbol[3:]  # Rest is quote

        # Order management
        self._orders: Dict[str, SimulatedOrder] = {}
        self._next_order_id = 1

        # Trading statistics
        self.stats = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "total_volume": 0.0,
            "total_fees": 0.0,
        }

    def connect(self) -> None:
        """Connect to Binance for market data."""
        try:
            self._binance.connect()
            self._status = ConnectionStatus.CONNECTED

            # Subscribe to order book updates for fill simulation
            self._binance.subscribe_orderbook(self._process_orderbook_update)

            print(f"[OK] Paper trader connected: {self.symbol}")
            print(f"[OK] Initial balances: {dict(self._balances)}")
        except Exception as e:
            self._status = ConnectionStatus.ERROR
            raise ConnectionError(f"Failed to connect paper trader: {e}")

    def disconnect(self) -> None:
        """Disconnect from Binance."""
        self._binance.disconnect()
        self._status = ConnectionStatus.DISCONNECTED
        print(f"[OK] Paper trader disconnected: {self.symbol}")
        print(f"[OK] Final balances: {dict(self._balances)}")
        print(f"[OK] Statistics: {self.stats}")

    def get_orderbook(self) -> OrderBookSnapshot:
        """Get current order book from Binance.

        Returns:
            Current order book state
        """
        return self._binance.get_orderbook()

    def submit_market_order(
        self,
        side: Side,
        quantity: Quantity,
        client_order_id: Optional[str] = None
    ) -> OrderResponse:
        """Submit simulated market order.

        Args:
            side: BUY or SELL
            quantity: Order quantity
            client_order_id: Optional client order ID

        Returns:
            OrderResponse with submission status
        """
        if self._status != ConnectionStatus.CONNECTED:
            return OrderResponse(
                success=False,
                message="Not connected",
                timestamp=Timestamp(time.time())
            )

        # Generate order ID
        order_id = self._generate_order_id()
        timestamp = Timestamp(time.time())

        # Create simulated order
        order = SimulatedOrder(
            order_id=order_id,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=None,
            timestamp=timestamp,
            client_order_id=client_order_id,
            status=OrderStatus.PENDING
        )

        self._orders[order_id] = order
        self.stats["orders_submitted"] += 1

        # Simulate immediate fill at current market price
        try:
            orderbook = self.get_orderbook()
            self._fill_market_order(order, orderbook)
        except Exception as e:
            return OrderResponse(
                success=False,
                message=f"Market order simulation failed: {e}",
                timestamp=timestamp,
                order_id=order_id,
                client_order_id=client_order_id
            )

        return OrderResponse(
            success=True,
            message="Market order filled",
            timestamp=timestamp,
            order_id=order_id,
            client_order_id=client_order_id
        )

    def submit_limit_order(
        self,
        side: Side,
        price: Price,
        quantity: Quantity,
        client_order_id: Optional[str] = None
    ) -> OrderResponse:
        """Submit simulated limit order.

        Args:
            side: BUY or SELL
            price: Limit price
            quantity: Order quantity
            client_order_id: Optional client order ID

        Returns:
            OrderResponse with submission status
        """
        if self._status != ConnectionStatus.CONNECTED:
            return OrderResponse(
                success=False,
                message="Not connected",
                timestamp=Timestamp(time.time())
            )

        # Generate order ID
        order_id = self._generate_order_id()
        timestamp = Timestamp(time.time())

        # Validate balance for buy orders
        if side == Side.BUY:
            required = float(price) * float(quantity)
            if self._balances[self.quote_asset] < required:
                return OrderResponse(
                    success=False,
                    message=f"Insufficient {self.quote_asset} balance",
                    timestamp=timestamp
                )
        else:
            if self._balances[self.base_asset] < float(quantity):
                return OrderResponse(
                    success=False,
                    message=f"Insufficient {self.base_asset} balance",
                    timestamp=timestamp
                )

        # Create simulated order
        order = SimulatedOrder(
            order_id=order_id,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            client_order_id=client_order_id,
            status=OrderStatus.ACTIVE
        )

        self._orders[order_id] = order
        self.stats["orders_submitted"] += 1

        # Reserve balance
        if side == Side.BUY:
            self._balances[self.quote_asset] -= float(price) * float(quantity)
        else:
            self._balances[self.base_asset] -= float(quantity)

        return OrderResponse(
            success=True,
            message="Limit order submitted",
            timestamp=timestamp,
            order_id=order_id,
            client_order_id=client_order_id
        )

    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel a simulated order.

        Args:
            order_id: Order ID to cancel

        Returns:
            OrderResponse with cancellation status
        """
        timestamp = Timestamp(time.time())

        if order_id not in self._orders:
            return OrderResponse(
                success=False,
                message=f"Order {order_id} not found",
                timestamp=timestamp
            )

        order = self._orders[order_id]

        if order.status not in (OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED):
            return OrderResponse(
                success=False,
                message=f"Order {order_id} cannot be cancelled (status: {order.status})",
                timestamp=timestamp
            )

        # Release reserved balance
        if order.order_type == OrderType.LIMIT:
            if order.side == Side.BUY:
                remaining_value = float(order.price) * float(order.remaining_quantity)
                self._balances[self.quote_asset] += remaining_value
            else:
                self._balances[self.base_asset] += float(order.remaining_quantity)

        order.status = OrderStatus.CANCELLED
        self.stats["orders_cancelled"] += 1

        return OrderResponse(
            success=True,
            message=f"Order {order_id} cancelled",
            timestamp=timestamp,
            order_id=order_id
        )

    def get_balance(self, asset: str) -> float:
        """Get current virtual balance.

        Args:
            asset: Asset symbol (e.g., 'BTC', 'USDT')

        Returns:
            Available balance
        """
        return self._balances[asset.upper()]

    def get_position(self) -> Dict[str, float]:
        """Get current positions across all assets.

        Returns:
            Dict of {asset: balance}
        """
        return dict(self._balances)

    def get_order_status(self, order_id: str) -> Optional[SimulatedOrder]:
        """Get status of a simulated order.

        Args:
            order_id: Order ID

        Returns:
            SimulatedOrder if found, None otherwise
        """
        return self._orders.get(order_id)

    # Private methods

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        order_id = f"PAPER_{self._next_order_id:08d}"
        self._next_order_id += 1
        return order_id

    def _fill_market_order(self, order: SimulatedOrder, orderbook: OrderBookSnapshot) -> None:
        """Simulate market order fill.

        Args:
            order: Market order to fill
            orderbook: Current order book snapshot
        """
        # Simulate latency
        if self.fill_latency_ms > 0:
            time.sleep(self.fill_latency_ms / 1000.0)

        # Determine fill price with slippage
        if order.side == Side.BUY:
            base_price = orderbook.best_ask[0] if orderbook.best_ask else orderbook.mid_price
        else:
            base_price = orderbook.best_bid[0] if orderbook.best_bid else orderbook.mid_price

        if base_price is None:
            raise RuntimeError("No market price available")

        # Apply slippage (worse for taker)
        slippage_factor = 1.0 + (self.slippage_bps / 10000.0)
        if order.side == Side.BUY:
            fill_price = Price(float(base_price) * slippage_factor)
        else:
            fill_price = Price(float(base_price) / slippage_factor)

        # Check balance
        if order.side == Side.BUY:
            required = float(fill_price) * float(order.quantity)
            if self._balances[self.quote_asset] < required:
                raise RuntimeError(f"Insufficient {self.quote_asset} balance")

            # Execute trade
            self._balances[self.quote_asset] -= required
            self._balances[self.base_asset] += float(order.quantity)
        else:
            if self._balances[self.base_asset] < float(order.quantity):
                raise RuntimeError(f"Insufficient {self.base_asset} balance")

            # Execute trade
            self._balances[self.base_asset] -= float(order.quantity)
            self._balances[self.quote_asset] += float(fill_price) * float(order.quantity)

        # Update order
        order.filled_quantity = order.quantity
        order.status = OrderStatus.FILLED

        # Create fill record
        fill = Fill(
            order_id=order.order_id,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=Timestamp(time.time()),
            trader_id="paper_trader",
            is_aggressor=True  # Market orders are always aggressive
        )

        # Notify subscribers
        self._notify_fill(fill)

        # Update stats
        self.stats["orders_filled"] += 1
        self.stats["total_volume"] += float(order.quantity)

    def _process_orderbook_update(self, orderbook: OrderBookSnapshot) -> None:
        """Process order book update and check for limit order fills.

        Args:
            orderbook: Updated order book snapshot
        """
        # Notify subscribers
        self._notify_orderbook_update(orderbook)

        # Check all active limit orders for fills
        for order in list(self._orders.values()):
            if order.status != OrderStatus.ACTIVE or order.order_type != OrderType.LIMIT:
                continue

            # Check if order would execute at current prices
            should_fill = False
            fill_price = order.price

            if order.side == Side.BUY:
                # Buy limit fills if ask <= limit price
                if orderbook.best_ask and orderbook.best_ask[0] <= order.price:
                    should_fill = True
                    fill_price = order.price  # Fill at limit or better
            else:
                # Sell limit fills if bid >= limit price
                if orderbook.best_bid and orderbook.best_bid[0] >= order.price:
                    should_fill = True
                    fill_price = order.price  # Fill at limit or better

            if should_fill:
                self._fill_limit_order(order, fill_price)

    def _fill_limit_order(self, order: SimulatedOrder, fill_price: Price) -> None:
        """Simulate limit order fill.

        Args:
            order: Limit order to fill
            fill_price: Fill price
        """
        # Simulate latency
        if self.fill_latency_ms > 0:
            time.sleep(self.fill_latency_ms / 1000.0)

        fill_quantity = order.remaining_quantity

        # Update balances (funds already reserved)
        if order.side == Side.BUY:
            self._balances[self.base_asset] += float(fill_quantity)
        else:
            self._balances[self.quote_asset] += float(fill_price) * float(fill_quantity)

        # Update order
        order.filled_quantity = Quantity(order.filled_quantity + fill_quantity)
        order.status = OrderStatus.FILLED if order.is_complete else OrderStatus.PARTIALLY_FILLED

        # Create fill record
        fill = Fill(
            order_id=order.order_id,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            timestamp=Timestamp(time.time()),
            trader_id="paper_trader",
            is_aggressor=False  # Limit orders are passive
        )

        # Notify subscribers
        self._notify_fill(fill)

        # Update stats
        if order.is_complete:
            self.stats["orders_filled"] += 1
        self.stats["total_volume"] += float(fill_quantity)
