"""SQLite-based state persistence for trading engine.

Manages durable storage of:
- Trade executions (fills)
- P&L history
- Position snapshots
- Strategy state
- Performance metrics

Database schema ensures data integrity and supports efficient queries
for analysis and recovery.
"""

import sqlite3
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
from .account_store import DurableAccountStore


class StateManager(DurableAccountStore):
    """Manages persistent state storage using SQLite.

    Provides atomic operations for storing and retrieving:
    - Trade executions
    - P&L snapshots
    - Position history
    - Strategy parameters

    Parameters:
    ----------
    db_path : str
        Path to SQLite database file
    """

    def __init__(self, db_path: str = "data/trading_state.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection_registry_lock = threading.RLock()
        self._connection_registry_condition = threading.Condition(self._connection_registry_lock)
        self._writer_connections = None
        self._writer_connections_closing = False
        self._connection_active = {}

        # Initialize database schema
        self._initialize_schema()
        self._initialize_account_store()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        identity = threading.get_ident()
        with self._connection_registry_condition:
            if self._writer_connections_closing:
                raise RuntimeError("paper writer connections are shutting down")
            registry = self._writer_connections
            reused = registry is not None
            if reused:
                conn = registry.get(identity)
                if conn is None:
                    conn = self._new_connection(cross_thread_close=True)
                    registry[identity] = conn
                self._connection_active[identity] = self._connection_active.get(identity, 0) + 1
            else:
                conn = self._new_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if reused:
                with self._connection_registry_condition:
                    self._connection_active[identity] -= 1
                    if not self._connection_active[identity]:
                        del self._connection_active[identity]
                    self._connection_registry_condition.notify_all()
            else:
                conn.close()

    def _new_connection(self, cross_thread_close=False):
        conn = sqlite3.connect(str(self.db_path), timeout=5.0,
                               check_same_thread=not cross_thread_close)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _initialize_schema(self) -> None:
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    order_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    trader_id TEXT NOT NULL,
                    is_aggressor INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create index on timestamp for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                ON trades(timestamp)
            """)

            # P&L history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pnl_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    inventory INTEGER NOT NULL,
                    cash REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    mid_price REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pnl_timestamp
                ON pnl_history(timestamp)
            """)

            # Positions table (current state)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL DEFAULT 'DEFAULT',
                    inventory INTEGER NOT NULL,
                    cash REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Strategy state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    strategy_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                ON metrics(timestamp)
            """)

            # Additive cross-asset paper journal. The legacy single-symbol
            # tables remain untouched for compatibility with existing runs.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cross_asset_events (
                    account_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (account_id, event_id)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cross_asset_events_time
                ON cross_asset_events(account_id, timestamp)
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cross_asset_snapshots (
                    account_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def record_trade(
        self,
        timestamp: float,
        order_id: str,
        side: str,
        quantity: int,
        price: float,
        trader_id: str,
        is_aggressor: bool
    ) -> None:
        """Record a trade execution.

        Parameters:
        ----------
        timestamp : float
            Trade timestamp
        order_id : str
            Order identifier
        side : str
            BUY or SELL
        quantity : int
            Trade size
        price : float
            Execution price
        trader_id : str
            Trader identifier
        is_aggressor : bool
            Whether this was the aggressing order
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades
                (timestamp, order_id, side, quantity, price, trader_id, is_aggressor)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, order_id, side, quantity, price, trader_id, int(is_aggressor)))

    def record_pnl(
        self,
        timestamp: float,
        inventory: int,
        cash: float,
        unrealized_pnl: float,
        realized_pnl: float,
        total_pnl: float,
        mid_price: float
    ) -> None:
        """Record P&L snapshot.

        Parameters:
        ----------
        timestamp : float
            Snapshot timestamp
        inventory : int
            Current inventory position
        cash : float
            Cash balance
        unrealized_pnl : float
            Unrealized P&L
        realized_pnl : float
            Realized P&L
        total_pnl : float
            Total P&L
        mid_price : float
            Current mid price
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pnl_history
                (timestamp, inventory, cash, unrealized_pnl, realized_pnl, total_pnl, mid_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, inventory, cash, unrealized_pnl, realized_pnl, total_pnl, mid_price))

    def update_position(
        self,
        timestamp: float,
        inventory: int,
        cash: float,
        avg_cost: float,
        unrealized_pnl: float,
        realized_pnl: float,
        symbol: str = "DEFAULT"
    ) -> None:
        """Update current position (upsert).

        Parameters:
        ----------
        timestamp : float
            Update timestamp
        inventory : int
            Current inventory
        cash : float
            Cash balance
        avg_cost : float
            Average cost basis
        unrealized_pnl : float
            Unrealized P&L
        realized_pnl : float
            Realized P&L
        symbol : str
            Symbol identifier
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if position exists
            cursor.execute("""
                SELECT id FROM positions WHERE symbol = ? ORDER BY id DESC LIMIT 1
            """, (symbol,))

            existing = cursor.fetchone()

            if existing:
                # Update existing
                cursor.execute("""
                    UPDATE positions
                    SET timestamp = ?, inventory = ?, cash = ?,
                        avg_cost = ?, unrealized_pnl = ?, realized_pnl = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (timestamp, inventory, cash, avg_cost, unrealized_pnl, realized_pnl, existing[0]))
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO positions
                    (timestamp, symbol, inventory, cash, avg_cost, unrealized_pnl, realized_pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, symbol, inventory, cash, avg_cost, unrealized_pnl, realized_pnl))

    def save_strategy_state(self, timestamp: float, strategy_id: str, state: Dict) -> None:
        """Save strategy state snapshot.

        Parameters:
        ----------
        timestamp : float
            Snapshot timestamp
        strategy_id : str
            Strategy identifier
        state : Dict
            Strategy state dictionary
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO strategy_state (timestamp, strategy_id, state_json)
                VALUES (?, ?, ?)
            """, (timestamp, strategy_id, json.dumps(state)))

    def record_metric(
        self,
        timestamp: float,
        metric_name: str,
        metric_value: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """Record performance metric.

        Parameters:
        ----------
        timestamp : float
            Metric timestamp
        metric_name : str
            Metric name
        metric_value : float
            Metric value
        metadata : Optional[Dict]
            Additional metadata
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else None
            cursor.execute("""
                INSERT INTO metrics (timestamp, metric_name, metric_value, metadata_json)
                VALUES (?, ?, ?, ?)
            """, (timestamp, metric_name, metric_value, metadata_json))

    def get_trades(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Retrieve trade history.

        Parameters:
        ----------
        start_time : Optional[float]
            Start timestamp filter
        end_time : Optional[float]
            End timestamp filter
        limit : int
            Maximum number of trades to return

        Returns:
        -------
        List[Dict]
            List of trade records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM trades WHERE 1=1"
            params = []

            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_pnl_history(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Retrieve P&L history.

        Parameters:
        ----------
        start_time : Optional[float]
            Start timestamp filter
        end_time : Optional[float]
            End timestamp filter
        limit : int
            Maximum number of records

        Returns:
        -------
        List[Dict]
            List of P&L snapshots
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM pnl_history WHERE 1=1"
            params = []

            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_current_position(self, symbol: str = "DEFAULT") -> Optional[Dict]:
        """Get current position for symbol.

        Parameters:
        ----------
        symbol : str
            Symbol identifier

        Returns:
        -------
        Optional[Dict]
            Current position or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM positions
                WHERE symbol = ?
                ORDER BY id DESC
                LIMIT 1
            """, (symbol,))

            row = cursor.fetchone()
            return dict(row) if row else None

    def get_latest_strategy_state(self, strategy_id: str) -> Optional[Dict]:
        """Get latest saved strategy state.

        Parameters:
        ----------
        strategy_id : str
            Strategy identifier

        Returns:
        -------
        Optional[Dict]
            Latest strategy state or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM strategy_state
                WHERE strategy_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (strategy_id,))

            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['state'] = json.loads(result['state_json'])
                return result
            return None

    def append_cross_asset_event(self, account_id: str, event_id: str, event_type: str,
                                 timestamp: float, payload: Dict) -> bool:
        """Append one idempotent event without touching legacy paper tables."""
        if not all(isinstance(value, str) and value.strip()
                   for value in (account_id, event_id, event_type)):
            raise ValueError("account_id, event_id, and event_type must be non-empty strings")
        stamp = float(timestamp)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO cross_asset_events
                (account_id, event_id, event_type, timestamp, payload_json)
                VALUES (?, ?, ?, ?, ?)
            """, (account_id, event_id, event_type, stamp, encoded))
            inserted = cursor.rowcount == 1
            if not inserted:
                cursor.execute("""
                    SELECT event_type, timestamp, payload_json FROM cross_asset_events
                    WHERE account_id = ? AND event_id = ?
                """, (account_id, event_id))
                existing = cursor.fetchone()
                if existing is None or (existing["event_type"] != event_type
                        or float(existing["timestamp"]) != stamp
                        or existing["payload_json"] != encoded):
                    raise ValueError("cross-asset event ID conflicts with previously stored data")
            return inserted

    def commit_cross_asset_event(self, account_id: str, event_id: str, event_type: str,
                                 timestamp: float, payload: Dict,
                                 snapshot: Dict,
                                 snapshot_timestamp: Optional[float] = None) -> bool:
        """Append a durable sequence-ordered transition and compact checkpoint."""
        return DurableAccountStore.commit_cross_asset_event(
            self, account_id, event_id, event_type, timestamp, payload,
            snapshot, snapshot_timestamp)

    def save_cross_asset_snapshot(self, account_id: str, timestamp: float,
                                  snapshot: Dict) -> None:
        return DurableAccountStore.save_cross_asset_snapshot(self, account_id, timestamp, snapshot)

    def load_cross_asset_snapshot(self, account_id: str) -> Optional[Dict]:
        return DurableAccountStore.load_cross_asset_snapshot(self, account_id)

    def get_cross_asset_events(self, account_id: str, limit: int = 100) -> List[Dict]:
        """Read recent cross-asset journal entries newest first."""
        count = max(0, int(limit))
        if count == 0:
            return []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT legacy.event_id, legacy.event_type, legacy.timestamp,
                       legacy.payload_json, journal.account_seq
                FROM cross_asset_events AS legacy
                LEFT JOIN paper_account_journal AS journal
                  ON journal.account_id = legacy.account_id
                 AND journal.event_id = legacy.event_id
                WHERE legacy.account_id = ?
                ORDER BY (journal.account_seq IS NOT NULL) DESC,
                         journal.account_seq DESC, legacy.rowid DESC LIMIT ?
            """, (account_id, count))
            return [{"event_id": row["event_id"], "event_type": row["event_type"],
                     "timestamp": float(row["timestamp"]),
                     "account_seq": row["account_seq"],
                     "payload": json.loads(row["payload_json"])} for row in cursor.fetchall()]

    def get_summary_stats(self) -> Dict:
        """Get summary statistics from database.

        Returns:
        -------
        Dict
            Summary statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Total trades
            cursor.execute("SELECT COUNT(*) as count FROM trades")
            stats['total_trades'] = cursor.fetchone()[0]

            # Latest P&L
            cursor.execute("""
                SELECT * FROM pnl_history
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            latest_pnl = cursor.fetchone()
            if latest_pnl:
                stats['latest_pnl'] = dict(latest_pnl)

            # Current position
            cursor.execute("""
                SELECT * FROM positions
                ORDER BY id DESC
                LIMIT 1
            """)
            current_pos = cursor.fetchone()
            if current_pos:
                stats['current_position'] = dict(current_pos)

            return stats

    def clear_all_data(self) -> None:
        """Clear all data from database (use with caution)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM trades")
            cursor.execute("DELETE FROM pnl_history")
            cursor.execute("DELETE FROM positions")
            cursor.execute("DELETE FROM strategy_state")
            cursor.execute("DELETE FROM metrics")
