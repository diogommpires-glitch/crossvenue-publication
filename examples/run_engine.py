"""Example: Run continuous trading engine with Avellaneda-Stoikov strategy.

This demonstrates the full trading engine runtime:
- Continuous event loop
- Real-time market simulation
- Strategy execution with order management
- State persistence (SQLite)
- Structured logging
- Health monitoring
- Graceful shutdown (Ctrl+C)

Usage:
    python examples/run_engine.py

    # Run for specific duration
    python examples/run_engine.py --duration 300  # 5 minutes

    # Custom parameters
    python examples/run_engine.py --risk-aversion 0.1 --order-size 200

Press Ctrl+C to gracefully stop the engine.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.market.market_simulator import MarketSimulator
from src.strategies.avellaneda_stoikov import AvellanedaStoikov
from src.runtime.trading_engine import TradingEngine
from src.runtime.state_manager import StateManager


def main():
    parser = argparse.ArgumentParser(
        description="Run continuous trading engine with market-making strategy"
    )

    # Engine parameters
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run duration in seconds (default: indefinite, stop with Ctrl+C)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory (default: logs)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/trading_state.db",
        help="SQLite database path (default: data/trading_state.db)"
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=60.0,
        help="Health heartbeat interval in seconds (default: 60.0)"
    )
    parser.add_argument(
        "--pnl-snapshot-interval",
        type=float,
        default=30.0,
        help="P&L snapshot interval in seconds (default: 30.0)"
    )

    # Market parameters
    parser.add_argument(
        "--initial-price",
        type=float,
        default=100.0,
        help="Initial mid price (default: 100.0)"
    )
    parser.add_argument(
        "--volatility",
        type=float,
        default=0.20,
        help="Annual volatility (default: 0.20)"
    )
    parser.add_argument(
        "--market-order-rate",
        type=float,
        default=1.0,
        help="Market orders per second (default: 1.0)"
    )
    parser.add_argument(
        "--limit-order-rate",
        type=float,
        default=5.0,
        help="Limit orders per second (default: 5.0)"
    )

    # Strategy parameters
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=0.05,
        help="Risk aversion parameter gamma (default: 0.05)"
    )
    parser.add_argument(
        "--order-size",
        type=int,
        default=100,
        help="Order size (default: 100)"
    )
    parser.add_argument(
        "--max-inventory",
        type=int,
        default=500,
        help="Maximum absolute inventory (default: 500)"
    )
    parser.add_argument(
        "--time-horizon",
        type=float,
        default=3600.0,
        help="Strategy time horizon in seconds (default: 3600.0 = 1 hour)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear existing database before starting"
    )

    args = parser.parse_args()

    # Create market simulator
    print("Initializing market simulator...")
    simulator = MarketSimulator(
        initial_price=args.initial_price,
        volatility=args.volatility,
        market_order_rate=args.market_order_rate,
        limit_order_rate=args.limit_order_rate,
        random_seed=args.seed
    )

    # Create strategy
    print("Initializing Avellaneda-Stoikov strategy...")
    strategy = AvellanedaStoikov(
        risk_aversion=args.risk_aversion,
        volatility=args.volatility,
        order_size=args.order_size,
        max_inventory=args.max_inventory,
        time_horizon=args.time_horizon,
        strategy_id="as_mm"
    )

    # Create state manager
    state_manager = StateManager(db_path=args.db_path)

    if args.clear_db:
        print(f"Clearing database: {args.db_path}")
        state_manager.clear_all_data()

    # Create trading engine
    print("Initializing trading engine...")
    engine = TradingEngine(
        strategy=strategy,
        simulator=simulator,
        state_manager=state_manager,
        log_dir=args.log_dir,
        db_path=args.db_path,
        heartbeat_interval=args.heartbeat_interval,
        pnl_snapshot_interval=args.pnl_snapshot_interval
    )

    # Print configuration
    print("\n" + "=" * 80)
    print("TRADING ENGINE CONFIGURATION")
    print("=" * 80)
    print(f"Duration: {'indefinite (Ctrl+C to stop)' if args.duration is None else f'{args.duration}s'}")
    print(f"Initial price: ${args.initial_price:.2f}")
    print(f"Volatility: {args.volatility:.2%}")
    print(f"Market order rate: {args.market_order_rate:.1f}/s")
    print(f"Limit order rate: {args.limit_order_rate:.1f}/s")
    print(f"Strategy: Avellaneda-Stoikov")
    print(f"  Risk aversion (γ): {args.risk_aversion:.3f}")
    print(f"  Order size: {args.order_size}")
    print(f"  Max inventory: ±{args.max_inventory}")
    print(f"  Time horizon: {args.time_horizon:.0f}s")
    print(f"Logging: {Path(args.log_dir).absolute()}")
    print(f"Database: {Path(args.db_path).absolute()}")
    print("=" * 80)
    print("\nStarting trading engine...")
    print("Press Ctrl+C to gracefully shutdown\n")

    # Run engine
    try:
        engine.run(duration=args.duration)
    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt caught (Ctrl+C)")
        print("Engine shutdown already handled by signal handler")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        raise

    print("\nEngine stopped. Check logs for details.")
    print(f"Logs: {Path(args.log_dir).absolute()}")
    print(f"Database: {Path(args.db_path).absolute()}")


if __name__ == "__main__":
    main()
