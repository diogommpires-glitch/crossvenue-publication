"""Continuous trading engine with event loop and state persistence.

This is the main runtime that orchestrates:
- Market simulation with real-time event processing
- Strategy execution and order management
- State persistence (trades, P&L, positions)
- Health monitoring and logging
- Graceful shutdown handling

The engine runs indefinitely until interrupted (Ctrl+C) and maintains
full state across restarts via SQLite persistence.
"""

import signal
import time
import logging
from typing import Optional, Dict, List
from pathlib import Path

from ..market.market_simulator import MarketSimulator, Event
from ..strategies.base import Strategy
from ..orderbook.order import Fill
from .state_manager import StateManager
from .logger import setup_logger, log_trade, log_pnl, log_health
from .health_monitor import HealthMonitor


class TradingEngine:
    """Continuous trading engine with event loop.

    Integrates market simulation, strategy execution, state persistence,
    and health monitoring into a unified runtime.

    State Recovery:
    --------------
    The engine saves strategy state on shutdown (inventory, cash, P&L).
    Recovery is opt-in via the `resume=True` parameter in run().
    Without resume, the engine starts fresh on each run.

    Parameters:
    ----------
    strategy : Strategy
        Trading strategy to execute
    simulator : MarketSimulator
        Market simulator for order flow generation
    state_manager : Optional[StateManager]
        State persistence manager (creates default if None)
    log_dir : str
        Directory for log files
    db_path : str
        Path to SQLite database
    heartbeat_interval : float
        Seconds between health heartbeats
    pnl_snapshot_interval : float
        Seconds between P&L snapshots
    """

    def __init__(
        self,
        strategy: Strategy,
        simulator: MarketSimulator,
        state_manager: Optional[StateManager] = None,
        log_dir: str = "logs",
        db_path: str = "data/trading_state.db",
        heartbeat_interval: float = 60.0,
        pnl_snapshot_interval: float = 30.0,
    ) -> None:
        self.strategy = strategy
        self.simulator = simulator
        self.state_manager = state_manager or StateManager(db_path)
        self.heartbeat_interval = heartbeat_interval
        self.pnl_snapshot_interval = pnl_snapshot_interval

        # Setup logging
        self.logger = setup_logger(
            name="trading_engine",
            log_dir=log_dir,
            console_level=logging.INFO,
            file_level=logging.DEBUG
        )

        # Setup health monitoring
        self.health_monitor = HealthMonitor(heartbeat_interval=heartbeat_interval)

        # Runtime state
        self.running = False
        self.shutdown_requested = False
        self.last_pnl_snapshot = time.time()

        # Statistics
        self.strategy_fills: List[Fill] = []
        self.total_strategy_orders = 0

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("=" * 80)
        self.logger.info("Trading Engine initialized")
        self.logger.info(f"Strategy: {self.strategy}")
        self.logger.info(f"Log directory: {Path(log_dir).absolute()}")
        self.logger.info(f"Database: {Path(db_path).absolute()}")
        self.logger.info("=" * 80)

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals (SIGINT, SIGTERM).

        Parameters:
        ----------
        signum : int
            Signal number
        frame : frame
            Current stack frame
        """
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        self.logger.info(f"\n{signal_name} received. Initiating graceful shutdown...")
        self.shutdown_requested = True

    def run(self, duration: Optional[float] = None, resume: bool = False) -> None:
        """Run the trading engine.

        Parameters:
        ----------
        duration : Optional[float]
            Run duration in seconds (None = indefinite)
        resume : bool
            If True, attempt to restore strategy state from last shutdown
        """
        self.running = True
        self.shutdown_requested = False

        self.logger.info("Starting trading engine...")
        self.logger.info(f"Run duration: {'indefinite' if duration is None else f'{duration}s'}")

        # Attempt state recovery if requested
        if resume:
            self.logger.info("Resume mode enabled, checking for saved state...")
            saved_state = self.state_manager.get_latest_strategy_state(
                self.strategy.strategy_id
            )
            if saved_state:
                state_data = saved_state['state']
                self.logger.info(f"Found saved state from {saved_state['created_at']}")
                self.logger.info(f"Restoring: inventory={state_data['inventory']}, "
                               f"cash={state_data['cash']:.2f}, "
                               f"realized_pnl={state_data['realized_pnl']:.2f}")

                # Restore strategy state
                self.strategy.inventory = state_data['inventory']
                self.strategy.cash = state_data['cash']
                self.strategy.realized_pnl = state_data['realized_pnl']
                if 'total_fills' in state_data:
                    self.strategy.total_fills = state_data['total_fills']

                self.logger.info("[OK] State restored successfully")
            else:
                self.logger.info("No saved state found, starting fresh")

        # Initialize market
        self.simulator.initialize_book(levels=5, orders_per_level=2)
        self.logger.info("Market initialized with liquidity")

        # Schedule organic order flow
        schedule_window = duration if duration is not None else 3600.0  # 1 hour default
        self.simulator.schedule_organic_orders(schedule_window)
        self.logger.info(f"Scheduled organic order flow for {schedule_window}s")

        # Record initial position
        self._snapshot_pnl()

        start_time = time.time()

        try:
            # Main event loop
            while self.running:
                # Check shutdown
                if self.shutdown_requested:
                    self.logger.info("Shutdown requested, stopping event loop...")
                    break

                # Check duration limit
                if duration is not None and (time.time() - start_time) >= duration:
                    self.logger.info(f"Duration limit reached ({duration}s), stopping...")
                    break

                # Process next market event
                event_start = time.time()
                event = self.simulator.step()

                if event is None:
                    # No more events, reschedule if running indefinitely
                    if duration is None:
                        self.logger.info("Event queue empty, scheduling more events...")
                        self.simulator.schedule_organic_orders(3600.0)
                        continue
                    else:
                        # Finite duration, we're done
                        self.logger.info("Event queue empty, simulation complete")
                        break

                # Record event processing time
                event_time_ms = (time.time() - event_start) * 1000.0
                self.health_monitor.record_event(event_time_ms)

                # Update strategy with market state
                self._update_strategy(event)

                # Periodic tasks
                self._periodic_tasks()

        except Exception as e:
            self.logger.error(f"Fatal error in event loop: {e}", exc_info=True)
            self.health_monitor.record_error(str(e), "fatal")
            raise
        finally:
            self._shutdown()

    def _update_strategy(self, event: Event) -> None:
        """Update strategy and process orders.

        Parameters:
        ----------
        event : Event
            Market event that was just processed
        """
        try:
            # Get current market state
            market_state = self.simulator.get_market_state()

            # Strategy generates orders
            orders = self.strategy.on_market_event(market_state)

            # Submit orders to market
            for order in orders:
                self.total_strategy_orders += 1
                final_order, fills = self.simulator.submit_strategy_order(order)

                # Process fills
                for fill in fills:
                    if fill.trader_id == self.strategy.strategy_id:
                        # This is our strategy's fill
                        self.strategy_fills.append(fill)
                        self.strategy.on_fill(fill, market_state["mid_price"])
                        self.health_monitor.record_fill()

                        # Log trade
                        log_trade(self.logger, {
                            "side": fill.side.name,
                            "quantity": fill.quantity,
                            "price": fill.price,
                            "order_id": fill.order_id,
                            "trader_id": fill.trader_id
                        })

                        # Persist trade
                        self.state_manager.record_trade(
                            timestamp=float(fill.timestamp),
                            order_id=fill.order_id,
                            side=fill.side.name,
                            quantity=fill.quantity,
                            price=float(fill.price),
                            trader_id=fill.trader_id,
                            is_aggressor=fill.is_aggressor
                        )

        except Exception as e:
            self.logger.error(f"Error updating strategy: {e}", exc_info=True)
            self.health_monitor.record_error(str(e), "strategy_update")

    def _periodic_tasks(self) -> None:
        """Execute periodic maintenance tasks."""
        current_time = time.time()

        # P&L snapshots
        if current_time - self.last_pnl_snapshot >= self.pnl_snapshot_interval:
            self._snapshot_pnl()
            self.last_pnl_snapshot = current_time

        # Health heartbeat
        if self.health_monitor.should_emit_heartbeat():
            heartbeat = self.health_monitor.emit_heartbeat()
            log_health(self.logger, heartbeat)

            # Update custom metrics
            market_state = self.simulator.get_market_state()
            self.health_monitor.set_custom_metric("mid_price", market_state["mid_price"])
            self.health_monitor.set_custom_metric("spread", market_state.get("spread", 0.0) or 0.0)
            self.health_monitor.set_custom_metric("inventory", self.strategy.inventory)

    def _snapshot_pnl(self) -> None:
        """Take P&L snapshot and persist."""
        try:
            market_state = self.simulator.get_market_state()
            mid_price = market_state["mid_price"]

            unrealized_pnl = self.strategy.get_unrealized_pnl(mid_price)
            total_pnl = self.strategy.get_total_pnl(mid_price)

            # Log P&L
            pnl_data = {
                "inventory": self.strategy.inventory,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": self.strategy.realized_pnl,
                "total_pnl": total_pnl
            }
            log_pnl(self.logger, pnl_data)

            # Persist P&L
            self.state_manager.record_pnl(
                timestamp=float(self.simulator.current_time),
                inventory=self.strategy.inventory,
                cash=self.strategy.cash,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=self.strategy.realized_pnl,
                total_pnl=total_pnl,
                mid_price=mid_price
            )

            # Update position
            avg_cost = -self.strategy.cash / self.strategy.inventory if self.strategy.inventory != 0 else 0.0
            self.state_manager.update_position(
                timestamp=float(self.simulator.current_time),
                inventory=self.strategy.inventory,
                cash=self.strategy.cash,
                avg_cost=avg_cost,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=self.strategy.realized_pnl
            )

        except Exception as e:
            self.logger.error(f"Error snapshotting P&L: {e}", exc_info=True)
            self.health_monitor.record_error(str(e), "pnl_snapshot")

    def _shutdown(self) -> None:
        """Graceful shutdown procedure."""
        self.logger.info("=" * 80)
        self.logger.info("Shutting down trading engine...")

        # Final P&L snapshot
        self.logger.info("Taking final P&L snapshot...")
        self._snapshot_pnl()

        # Save strategy state
        self.logger.info("Saving strategy state...")
        try:
            market_state = self.simulator.get_market_state()
            strategy_state = self.strategy.get_state(market_state["mid_price"])
            self.state_manager.save_strategy_state(
                timestamp=float(self.simulator.current_time),
                strategy_id=self.strategy.strategy_id,
                state={
                    "inventory": strategy_state.inventory,
                    "cash": strategy_state.cash,
                    "unrealized_pnl": strategy_state.unrealized_pnl,
                    "realized_pnl": strategy_state.realized_pnl,
                    "total_fills": strategy_state.total_fills
                }
            )
        except Exception as e:
            self.logger.error(f"Error saving strategy state: {e}", exc_info=True)

        # Final health status
        final_health = self.health_monitor.get_status()
        log_health(self.logger, final_health)

        # Summary statistics
        self.logger.info("=" * 80)
        self.logger.info("FINAL SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Uptime: {self.health_monitor.metrics.uptime_formatted()}")
        self.logger.info(f"Total events processed: {self.health_monitor.metrics.total_events}")
        self.logger.info(f"Strategy orders submitted: {self.total_strategy_orders}")
        self.logger.info(f"Strategy fills: {len(self.strategy_fills)}")
        self.logger.info(f"Final inventory: {self.strategy.inventory}")
        self.logger.info(f"Final cash: {self.strategy.cash:.2f}")

        market_state = self.simulator.get_market_state()
        final_pnl = self.strategy.get_total_pnl(market_state["mid_price"])
        self.logger.info(f"Final P&L: {final_pnl:.2f}")
        self.logger.info(f"Errors: {self.health_monitor.metrics.error_count}")
        self.logger.info(f"Warnings: {self.health_monitor.metrics.warning_count}")
        self.logger.info("=" * 80)

        # Get database summary
        try:
            db_stats = self.state_manager.get_summary_stats()
            self.logger.info(f"Database stats: {db_stats.get('total_trades', 0)} total trades recorded")
        except Exception as e:
            self.logger.error(f"Error retrieving DB stats: {e}")

        self.running = False
        self.logger.info("Trading engine shutdown complete")
        self.logger.info("=" * 80)

    def get_status(self) -> Dict:
        """Get current engine status.

        Returns:
        -------
        Dict
            Current status dictionary
        """
        market_state = self.simulator.get_market_state()
        health_status = self.health_monitor.get_status()

        return {
            "running": self.running,
            "uptime": health_status["uptime_formatted"],
            "market_time": float(self.simulator.current_time),
            "mid_price": market_state["mid_price"],
            "spread": market_state.get("spread", 0.0),
            "strategy_inventory": self.strategy.inventory,
            "strategy_cash": self.strategy.cash,
            "strategy_pnl": self.strategy.get_total_pnl(market_state["mid_price"]),
            "total_fills": len(self.strategy_fills),
            "health": health_status,
        }
