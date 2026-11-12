"""Tests for P&L tracking and metrics."""

import pytest
import numpy as np
from src.metrics.pnl import PnLTracker, PnLSnapshot
from src.utils.types import Timestamp


def test_sharpe_ratio_zero_volatility():
    """Test Sharpe ratio with constant returns."""
    tracker = PnLTracker(starting_capital=10000.0)

    # Constant return of $10 per period
    for i in range(100):
        timestamp = Timestamp(i * 60.0)  # Every 60 seconds
        pnl = i * 10.0  # Constant $10 growth

        snapshot = PnLSnapshot(
            timestamp=timestamp,
            cash=10000.0,
            inventory=0.0,
            realized_pnl=pnl,
            unrealized_pnl=0.0,
            total_pnl=pnl
        )
        tracker.snapshots.append(snapshot)

    # With constant returns, vol should be ~0, Sharpe undefined
    sharpe = tracker.calculate_sharpe_ratio()

    # Should return NaN, 0, or inf for zero variance (depends on implementation)
    assert np.isnan(sharpe) or sharpe == 0.0 or np.isinf(sharpe)


def test_sharpe_ratio_realistic_returns():
    """Test Sharpe ratio with realistic volatile returns."""
    np.random.seed(42)
    tracker = PnLTracker(starting_capital=10000.0)

    # Simulate returns: mean 0.1% per minute, vol 0.5%
    cumulative_pnl = 0.0
    for i in range(1000):  # ~16 hours of trading
        timestamp = Timestamp(i * 60.0)

        # Random return
        ret = np.random.normal(0.001, 0.005)  # Mean 0.1%, std 0.5%
        cumulative_pnl += ret * 10000.0

        snapshot = PnLSnapshot(
            timestamp=timestamp,
            cash=10000.0,
            inventory=0.0,
            realized_pnl=cumulative_pnl,
            unrealized_pnl=0.0,
            total_pnl=cumulative_pnl
        )
        tracker.snapshots.append(snapshot)

    sharpe = tracker.calculate_sharpe_ratio()

    # With mean 0.1% and vol 0.5% per minute:
    # Annual mean ≈ 0.001 × 525,600 min/year = 525.6 (52,560%)
    # Annual vol ≈ 0.005 × sqrt(525,600) ≈ 3.62 (362%)
    # Sharpe ≈ 525.6 / 3.62 ≈ 145

    # Allow wide range due to randomness
    assert 50 < sharpe < 250, f"Sharpe {sharpe} outside expected range"


def test_sharpe_ratio_negative_returns():
    """Test Sharpe ratio correctly handles losing strategy."""
    tracker = PnLTracker(starting_capital=10000.0)

    # Losing money steadily
    for i in range(100):
        timestamp = Timestamp(i * 60.0)
        pnl = -i * 10.0  # Losing $10 per period

        snapshot = PnLSnapshot(
            timestamp=timestamp,
            cash=10000.0,
            inventory=0.0,
            realized_pnl=pnl,
            unrealized_pnl=0.0,
            total_pnl=pnl
        )
        tracker.snapshots.append(snapshot)

    sharpe = tracker.calculate_sharpe_ratio()

    # Should be negative (losing money)
    assert sharpe < 0, "Losing strategy should have negative Sharpe"
