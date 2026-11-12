"""Abstract base connector interface for market data and execution.

Defines the contract that all connectors (live, paper, etc.) must implement.
This abstraction allows strategies to be exchange-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from enum import Enum

from ..utils.types import Side, OrderType, Price, Quantity, Timestamp
from ..orderbook.order import Order, Fill


class ConnectionStatus(Enum):
    """Connector connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class OrderBookSnapshot:
    """Order book snapshot at a point in time.

    Attributes:
        timestamp: Snapshot timestamp (exchange time)
        bids: List of (price, quantity) tuples, sorted descending by price
        asks: List of (price, quantity) tuples, sorted ascending by price
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
    """
    timestamp: Timestamp
    bids: List[tuple[Price, Quantity]]
    asks: List[tuple[Price, Quantity]]
    symbol: str

    @property
    def best_bid(self) -> Optional[tuple[Price, Quantity]]:
        """Best bid (highest buy price)."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[tuple[Price, Quantity]]:
        """Best ask (lowest sell price)."""
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> Optional[Price]:
        """Mid price between best bid and ask."""
        if self.best_bid and self.best_ask:
            return Price((self.best_bid[0] + self.best_ask[0]) / 2.0)
        return None

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread in price units."""
        if self.best_bid and self.best_ask:
            return float(self.best_ask[0] - self.best_bid[0])
        return None


@dataclass
class OrderResponse:
    """Response from order submission.

    Attributes:
        success: Whether order was accepted
        order_id: Exchange-assigned order ID (if successful)
        message: Human-readable status message
        timestamp: Response timestamp
        client_order_id: Client-side order ID (optional)
    """
    success: bool
    message: str
    timestamp: Timestamp
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None


class BaseConnector(ABC):
    """Abstract base class for exchange connectors.

    All connectors must implement these methods to provide:
    - Real-time order book data via WebSocket
    - Order execution via REST API
    - Connection management and error handling
    """

    def __init__(self, symbol: str, config: Optional[Dict] = None):
        """Initialize connector.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            config: Connector-specific configuration
        """
        self.symbol = symbol
        self.config = config or {}
        self._status = ConnectionStatus.DISCONNECTED
        self._orderbook_callbacks: List[Callable[[OrderBookSnapshot], None]] = []
        self._fill_callbacks: List[Callable[[Fill], None]] = []

    @property
    def status(self) -> ConnectionStatus:
        """Current connection status."""
        return self._status

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to exchange.

        This should:
        - Authenticate with API credentials
        - Open WebSocket connections
        - Start receiving market data

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close all connections gracefully."""
        pass

    @abstractmethod
    def get_orderbook(self) -> OrderBookSnapshot:
        """Get current order book snapshot.

        Returns:
            Current order book state

        Raises:
            RuntimeError: If not connected
        """
        pass

    @abstractmethod
    def submit_market_order(
        self,
        side: Side,
        quantity: Quantity,
        client_order_id: Optional[str] = None
    ) -> OrderResponse:
        """Submit market order.

        Args:
            side: BUY or SELL
            quantity: Order quantity
            client_order_id: Optional client-side order ID

        Returns:
            OrderResponse with submission status
        """
        pass

    @abstractmethod
    def submit_limit_order(
        self,
        side: Side,
        price: Price,
        quantity: Quantity,
        client_order_id: Optional[str] = None
    ) -> OrderResponse:
        """Submit limit order.

        Args:
            side: BUY or SELL
            price: Limit price
            quantity: Order quantity
            client_order_id: Optional client-side order ID

        Returns:
            OrderResponse with submission status
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an active order.

        Args:
            order_id: Exchange order ID to cancel

        Returns:
            OrderResponse with cancellation status
        """
        pass

    @abstractmethod
    def get_balance(self, asset: str) -> float:
        """Get current balance for an asset.

        Args:
            asset: Asset symbol (e.g., 'BTC', 'USDT')

        Returns:
            Available balance
        """
        pass

    def subscribe_orderbook(self, callback: Callable[[OrderBookSnapshot], None]) -> None:
        """Subscribe to order book updates.

        Args:
            callback: Function to call on each update
        """
        self._orderbook_callbacks.append(callback)

    def subscribe_fills(self, callback: Callable[[Fill], None]) -> None:
        """Subscribe to fill notifications.

        Args:
            callback: Function to call on each fill
        """
        self._fill_callbacks.append(callback)

    def _notify_orderbook_update(self, snapshot: OrderBookSnapshot) -> None:
        """Notify all subscribers of order book update."""
        for callback in self._orderbook_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                print(f"Error in orderbook callback: {e}")

    def _notify_fill(self, fill: Fill) -> None:
        """Notify all subscribers of fill."""
        for callback in self._fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                print(f"Error in fill callback: {e}")
