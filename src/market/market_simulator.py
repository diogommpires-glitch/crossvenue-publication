"""Market simulator with synthetic order flow generation.

This module simulates realistic market dynamics including:
- Stochastic mid-price evolution
- Poisson arrival of market orders
- Organic limit order flow
- Integration with self-impact model
"""

import heapq
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from ..orderbook.limit_order_book import LimitOrderBook
from ..orderbook.order import Order, Fill
from ..utils.types import Side, OrderType, Price, Quantity, Timestamp
from .self_impact import SelfImpactModel


class EventType(Enum):
    """Types of market events."""

    MARKET_ORDER = auto()
    LIMIT_ORDER = auto()
    CANCEL_ORDER = auto()
    PRICE_UPDATE = auto()


@dataclass(order=True)
class Event:
    """Discrete event for event-driven simulation.

    Events are ordered by timestamp for priority queue processing.
    """

    timestamp: Timestamp
    event_type: EventType = None
    data: dict = None

    def __post_init__(self):
        # Ensure proper ordering in heapq (only timestamp matters)
        object.__setattr__(self, 'event_type', self.event_type)
        object.__setattr__(self, 'data', self.data or {})


class MarketSimulator:
    """Event-driven market simulator with realistic order flow.

    This simulator generates synthetic but realistic market conditions:
    - Mid-price follows geometric Brownian motion with drift
    - Market orders arrive via Poisson process
    - Limit orders placed around mid-price
    - Market impact from large trades

    The simulator integrates with SelfImpactModel to track strategy impact
    separately from natural market movement.

    Parameters:
    ----------
    initial_price : float
        Starting mid-price
    volatility : float
        Annual volatility (for GBM price process)
    drift : float
        Annual drift (for GBM price process)
    market_order_rate : float
        Average market orders per second (Poisson rate)
    limit_order_rate : float
        Average limit orders per second (Poisson rate)
    typical_order_size : int
        Mean order size (Poisson distributed)
    spread_mean : float
        Average bid-ask spread
    random_seed : Optional[int]
        Random seed for reproducibility
    impact_coefficient_bps : float
        Log-price response in basis points for one reference-depth aggressive order
    impact_reference_quantity : Optional[float]
        Fallback reference depth when the visible book is empty
    impact_decay_halflife : float
        Exponential impact half-life in simulation seconds
    impact_lookback_window : float
        Retained impact and fill history in simulation seconds
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        volatility: float = 0.20,  # 20% annual vol
        drift: float = 0.0,  # No drift
        market_order_rate: float = 1.0,  # 1 market order/sec
        limit_order_rate: float = 5.0,  # 5 limit orders/sec
        typical_order_size: int = 100,
        spread_mean: float = 0.10,
        impact_enabled: bool = False,
        random_seed: Optional[int] = None,
        impact_coefficient_bps: float = 5.0,
        impact_reference_quantity: Optional[float] = None,
        impact_decay_halflife: float = 30.0,
        impact_lookback_window: float = 300.0,
        seconds_per_year: float = 365.25 * 24 * 3600,
    ) -> None:
        if not math.isfinite(initial_price) or initial_price <= 0:
            raise ValueError("initial_price must be finite and positive")
        if not math.isfinite(volatility) or volatility < 0:
            raise ValueError("volatility must be finite and non-negative")
        if not math.isfinite(drift):
            raise ValueError("drift must be finite")
        if not math.isfinite(market_order_rate) or market_order_rate <= 0:
            raise ValueError("market_order_rate must be finite and positive")
        if not math.isfinite(limit_order_rate) or limit_order_rate <= 0:
            raise ValueError("limit_order_rate must be finite and positive")
        if not isinstance(typical_order_size, int) or typical_order_size <= 0:
            raise ValueError("typical_order_size must be a positive integer")
        if not math.isfinite(spread_mean) or spread_mean <= 0:
            raise ValueError("spread_mean must be finite and positive")
        self.initial_price = initial_price
        self.volatility = volatility
        self.drift = drift
        self.market_order_rate = market_order_rate
        self.limit_order_rate = limit_order_rate
        self.typical_order_size = typical_order_size
        self.spread_mean = spread_mean
        self.impact_enabled = impact_enabled

        # Convert annual volatility to per-second for GBM
        # σ_annual = σ_second * sqrt(seconds_per_year)
        self.seconds_per_year = float(seconds_per_year)
        if not math.isfinite(self.seconds_per_year) or self.seconds_per_year <= 0:
            raise ValueError("seconds_per_year must be finite and positive")
        self.vol_per_second = self.volatility / np.sqrt(self.seconds_per_year)
        self.drift_per_second = self.drift / self.seconds_per_year

        # Random state
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # Market state
        self.book = LimitOrderBook()
        self.current_time: Timestamp = Timestamp(0.0)
        self.base_mid_price: Price = Price(initial_price)
        self.mid_price: Price = Price(initial_price)

        # Impact is calibrated in basis points at a reference quantity. The
        # simulator passes pre-trade displayed depth when it is available.
        self.impact_reference_quantity = float(
            typical_order_size
            if impact_reference_quantity is None
            else impact_reference_quantity
        )
        self.self_impact = SelfImpactModel(
            impact_coefficient_bps=impact_coefficient_bps,
            reference_quantity=self.impact_reference_quantity,
            decay_halflife=impact_decay_halflife,
            strategy_id="mm_strategy",
            lookback_window=impact_lookback_window,
        )

        # Event queue (min-heap)
        self.event_queue: List[Event] = []

        # Statistics
        self.total_events: int = 0
        self.organic_fills: List[Fill] = []

        # Price update tracking (FIX: for correct GBM timing)
        self.last_price_update: Optional[Timestamp] = None

        # Strategy callback (set by strategy)
        self.strategy_callback: Optional[Callable] = None

    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.book = LimitOrderBook()
        self.current_time = Timestamp(0.0)
        self.base_mid_price = Price(self.initial_price)
        self.mid_price = Price(self.initial_price)
        self.event_queue.clear()
        self.self_impact.reset()
        self.rng = np.random.RandomState(self.random_seed)
        self.total_events = 0
        self.organic_fills.clear()
        self.last_price_update = None

    def initialize_book(self, levels: int = 5, orders_per_level: int = 2) -> None:
        """Initialize order book with resting liquidity.

        Creates a realistic starting state with bid/ask levels.
        """
        half_spread = self.spread_mean / 2.0

        # Add bids
        for i in range(levels):
            price_offset = half_spread + i * 0.05
            for j in range(orders_per_level):
                bid = Order(
                    side=Side.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=Quantity(self.typical_order_size),
                    price=Price(self.mid_price - price_offset),
                    timestamp=self.current_time,
                    trader_id=f"organic_bid_{i}_{j}",
                )
                self.book.add_order(bid)

        # Add asks
        for i in range(levels):
            price_offset = half_spread + i * 0.05
            for j in range(orders_per_level):
                ask = Order(
                    side=Side.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=Quantity(self.typical_order_size),
                    price=Price(self.mid_price + price_offset),
                    timestamp=self.current_time,
                    trader_id=f"organic_ask_{i}_{j}",
                )
                self.book.add_order(ask)

    def schedule_organic_orders(self, duration: float) -> None:
        """Schedule random organic order flow for duration.

        Uses Poisson processes for arrival times.
        """
        # Generate market order arrivals
        market_arrivals = self._generate_poisson_arrivals(
            self.market_order_rate, duration
        )
        for arrival_time in market_arrivals:
            event = Event(
                timestamp=Timestamp(arrival_time),
                event_type=EventType.MARKET_ORDER,
                data={},
            )
            heapq.heappush(self.event_queue, event)

        # Generate limit order arrivals
        limit_arrivals = self._generate_poisson_arrivals(
            self.limit_order_rate, duration
        )
        for arrival_time in limit_arrivals:
            event = Event(
                timestamp=Timestamp(arrival_time),
                event_type=EventType.LIMIT_ORDER,
                data={},
            )
            heapq.heappush(self.event_queue, event)

    def step(self) -> Optional[Event]:
        """Process next event in the queue.

        Returns:
            The processed event, or None if queue is empty
        """
        if not self.event_queue:
            return None

        event = heapq.heappop(self.event_queue)
        self.current_time = event.timestamp
        self.total_events += 1

        # Update mid-price (GBM with impact)
        self._update_mid_price(event)

        # Process event
        if event.event_type == EventType.MARKET_ORDER:
            self._process_market_order(event)
        elif event.event_type == EventType.LIMIT_ORDER:
            self._process_limit_order(event)
        elif event.event_type == EventType.CANCEL_ORDER:
            self._process_cancel_order(event)

        # Notify strategy callback
        if self.strategy_callback is not None:
            self.strategy_callback(event)

        return event

    def run(self, duration: float, strategy_callback: Optional[Callable] = None) -> None:
        """Run simulation for specified duration.

        Parameters:
        ----------
        duration : float
            Simulation duration in seconds
        strategy_callback : Optional[Callable]
            Function called after each event with (simulator, event)
        """
        self.strategy_callback = strategy_callback
        self.schedule_organic_orders(duration)

        while self.event_queue:
            event = self.step()
            if event is None:
                break

    def submit_strategy_order(self, order: Order) -> Tuple[Order, List[Fill]]:
        """Submit a strategy order and attribute aggressive impact to its trader ID."""
        self.self_impact.strategy_id = order.trader_id
        reference_quantity = self._reference_depth(order.side)
        final_order, fills = self.book.add_order(order)
        for fill in fills:
            self.self_impact.record_fill(fill, self.current_time, reference_quantity=reference_quantity)
            if fill.trader_id != self.self_impact.strategy_id:
                self.organic_fills.append(fill)
        self._compose_mid_price()
        return final_order, fills


    def get_market_state(self) -> Dict:
        """Return the exogenous price, composed price, and impact attribution."""
        self.self_impact.advance_time(self.current_time)
        self._compose_mid_price()
        return {
            "time": self.current_time,
            "seconds_per_year": self.seconds_per_year,
            "base_mid_price": self.base_mid_price,
            "mid_price": self.mid_price,
            # Compatibility key; this aliases the composed price.
            "impact_adjusted_mid": self.mid_price,
            "best_bid": self.book.get_best_bid(),
            "best_ask": self.book.get_best_ask(),
            "spread": self.book.get_spread(),
            "cumulative_impact": self.self_impact.get_cumulative_impact(),
            "strategy_impact": self.self_impact.get_strategy_impact(),
            "market_impact": self.self_impact.get_market_impact(),
        }

    def _compose_mid_price(self) -> None:
        """Update the impacted reference mid without moving resting executable quotes."""
        if self.impact_enabled:
            target_mid = float(self.self_impact.get_impact_adjusted_price(self.base_mid_price))
        else:
            target_mid = float(self.base_mid_price)
        if not math.isfinite(target_mid) or target_mid <= 0:
            raise ValueError("composed mid-price must be finite and positive")
        self.mid_price = Price(target_mid)


    def _reference_depth(self, aggressive_side: Side) -> float:
        """Read visible depth on the side an incoming order will consume."""
        resting_side = Side.SELL if aggressive_side == Side.BUY else Side.BUY
        depth = sum(
            float(quantity)
            for _, quantity in self.book.get_depth(resting_side, levels=5)
        )
        if depth > 0 and np.isfinite(depth):
            return depth
        return self.impact_reference_quantity

    def _generate_poisson_arrivals(self, rate: float, duration: float) -> List[float]:
        """Generate Poisson arrival times."""
        arrivals = []
        t = 0.0

        while t < duration:
            # Inter-arrival time is exponentially distributed
            inter_arrival = self.rng.exponential(1.0 / rate)
            t += inter_arrival
            if t < duration:
                arrivals.append(t)

        return arrivals

    def _update_mid_price(self, event: Event) -> None:
        """Advance exact log-GBM over the full elapsed interval, then apply impact."""
        if not math.isfinite(float(event.timestamp)) or float(event.timestamp) < 0:
            raise ValueError("event timestamp must be finite and non-negative")
        self.self_impact.advance_time(self.current_time)
        if self.last_price_update is None:
            self.last_price_update = self.current_time
            self._compose_mid_price()
            return
        dt = float(self.current_time - self.last_price_update)
        if dt <= 0:
            self._compose_mid_price()
            return
        sigma = float(self.vol_per_second)
        mu = float(self.drift_per_second)
        log_return = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * float(self.rng.randn())
        self.base_mid_price = Price(float(self.base_mid_price) * math.exp(log_return))
        self.last_price_update = self.current_time
        self._compose_mid_price()


    def _process_market_order(self, event: Event) -> None:
        """Process an organic market order."""
        # Random side and size
        side = Side.BUY if self.rng.rand() < 0.5 else Side.SELL
        quantity = max(1, int(self.rng.poisson(self.typical_order_size)))

        order = Order(
            side=side,
            order_type=OrderType.MARKET,
            quantity=Quantity(quantity),
            price=None,
            timestamp=self.current_time,
            trader_id=f"organic_market_{self.total_events}",
        )

        reference_quantity = self._reference_depth(order.side)
        final_order, fills = self.book.add_order(order)

        # Track fills
        for fill in fills:
            self.self_impact.record_fill(
                fill,
                self.current_time,
                reference_quantity=reference_quantity,
            )
            if fill.trader_id != "mm_strategy":
                self.organic_fills.append(fill)

        self._compose_mid_price()

    def _process_limit_order(self, event: Event) -> None:
        """Process an organic limit order."""
        # Random side
        side = Side.BUY if self.rng.rand() < 0.5 else Side.SELL

        # Random size
        quantity = max(1, int(self.rng.poisson(self.typical_order_size)))

        # Price slightly away from mid (realistic limit order placement)
        offset = self.rng.exponential(self.spread_mean)
        if side == Side.BUY:
            price = Price(max(0.01, self.mid_price - offset))
        else:
            price = Price(self.mid_price + offset)

        order = Order(
            side=side,
            order_type=OrderType.LIMIT,
            quantity=Quantity(quantity),
            price=price,
            timestamp=self.current_time,
            trader_id=f"organic_limit_{self.total_events}",
        )

        reference_quantity = self._reference_depth(order.side)
        final_order, fills = self.book.add_order(order)

        # Track fills
        for fill in fills:
            self.self_impact.record_fill(
                fill,
                self.current_time,
                reference_quantity=reference_quantity,
            )
            if fill.trader_id != "mm_strategy":
                self.organic_fills.append(fill)

        self._compose_mid_price()

    def _process_cancel_order(self, event: Event) -> None:
        """Process an order cancellation.

        Cancels an order by order_id specified in event.data['order_id'].
        Removes the order from the book and all tracking structures.
        """
        order_id = event.data.get('order_id')
        if order_id is None:
            return

        # Cancel the order through the book's interface
        success = self.book.cancel_order(order_id)

        if not success:
            # Order may have already been filled or cancelled
            pass

    def __str__(self) -> str:
        state = self.get_market_state()
        return (
            f"Market(t={state['time']:.1f}, mid={state['mid_price']:.2f}, "
            f"spread={state['spread']}, events={self.total_events})"
        )
