"""Test Sharpe ratio calculation in PnLTracker.

Validates that:
- Constant returns give high Sharpe ratio
- Losing strategy gives negative Sharpe ratio
- Calculation uses starting capital base, not P&L
"""

import numpy as np
import pytest

from src.metrics.pnl import PnLTracker, PnLSnapshot
from src.utils.types import Timestamp


class TestSharpeRatioCalculation:
    """Test Sharpe ratio calculation correctness."""

    def test_constant_returns_high_sharpe(self):
        """Constant positive returns should give infinite Sharpe ratio."""
        tracker = PnLTracker(
            starting_capital=10000.0,
            snapshot_interval=60.0,
        )

        # Manually create snapshots with constant returns
        # Each snapshot increases P&L by 1% of starting capital ($100)
        base_time = 1000.0
        constant_return = 100.0  # 1% of 10000

        for i in range(50):
            snapshot = PnLSnapshot(
                timestamp=Timestamp(base_time + i * 60.0),
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                total_pnl=constant_return * i,  # Linear growth
                inventory=0,
                cash=constant_return * i,
            )
            tracker.snapshots.append(snapshot)

        sharpe = tracker.calculate_sharpe_ratio()

        # Constant positive returns with zero variance should give infinite Sharpe
        assert np.isinf(sharpe), f"Expected infinite Sharpe for constant positive returns, got {sharpe}"
        assert sharpe > 0, "Sharpe should be positive infinity"

    def test_losing_strategy_negative_sharpe(self):
        """Strategy with consistent losses should give negative infinite Sharpe ratio."""
        tracker = PnLTracker(
            starting_capital=10000.0,
            snapshot_interval=60.0,
        )

        # Create snapshots with consistent losses
        # Each snapshot decreases P&L by 0.5% of starting capital ($50)
        base_time = 1000.0
        loss_per_period = -50.0

        for i in range(50):
            snapshot = PnLSnapshot(
                timestamp=Timestamp(base_time + i * 60.0),
                realized_pnl=loss_per_period * i,
                unrealized_pnl=0.0,
                total_pnl=loss_per_period * i,
                inventory=0,
                cash=loss_per_period * i,
            )
            tracker.snapshots.append(snapshot)

        sharpe = tracker.calculate_sharpe_ratio()

        # Constant negative returns with zero variance should give negative infinite Sharpe
        assert np.isinf(sharpe), f"Expected infinite Sharpe for constant losses, got {sharpe}"
        assert sharpe < 0, "Sharpe should be negative infinity"

    def test_uses_starting_capital_not_pnl(self):
        """Verify that returns are calculated using starting capital, not P&L."""
        tracker = PnLTracker(
            starting_capital=10000.0,
            snapshot_interval=60.0,
        )

        # Create snapshots with increasing P&L
        base_time = 1000.0
        pnl_values = [0, 100, 250, 350, 500]  # Varying increases

        for i, pnl in enumerate(pnl_values):
            snapshot = PnLSnapshot(
                timestamp=Timestamp(base_time + i * 60.0),
                realized_pnl=pnl,
                unrealized_pnl=0.0,
                total_pnl=pnl,
                inventory=0,
                cash=pnl,
            )
            tracker.snapshots.append(snapshot)

        # Manually calculate expected returns using starting capital
        pnls = np.array(pnl_values)
        equity_changes = np.diff(pnls)  # [100, 150, 100, 150]
        expected_returns = equity_changes / tracker.starting_capital
        expected_mean = np.mean(expected_returns)
        expected_std = np.std(expected_returns, ddof=1)

        # Calculate Sharpe
        sharpe = tracker.calculate_sharpe_ratio()

        # Verify the calculation manually
        if expected_std > 0:
            expected_sharpe_raw = expected_mean / expected_std
            periods_per_year = (365.25 * 24 * 3600) / tracker.snapshot_interval
            expected_sharpe = expected_sharpe_raw * np.sqrt(periods_per_year)

            assert not np.isnan(sharpe), "Sharpe should not be NaN"
            assert np.isclose(sharpe, expected_sharpe, rtol=1e-10), \
                f"Sharpe calculation mismatch: got {sharpe}, expected {expected_sharpe}"

    def test_insufficient_data_returns_nan(self):
        """Sharpe ratio should return NaN with insufficient data."""
        tracker = PnLTracker(starting_capital=10000.0)

        # No snapshots
        assert np.isnan(tracker.calculate_sharpe_ratio())

        # Only one snapshot
        snapshot = PnLSnapshot(
            timestamp=Timestamp(1000.0),
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_pnl=0.0,
            inventory=0,
            cash=0.0,
        )
        tracker.snapshots.append(snapshot)
        assert np.isnan(tracker.calculate_sharpe_ratio())

    def test_zero_variance_returns_nan(self):
        """Sharpe ratio should return NaN when all returns are identical (zero variance)."""
        tracker = PnLTracker(starting_capital=10000.0)

        # Create snapshots with no change in P&L
        base_time = 1000.0
        for i in range(10):
            snapshot = PnLSnapshot(
                timestamp=Timestamp(base_time + i * 60.0),
                realized_pnl=100.0,
                unrealized_pnl=0.0,
                total_pnl=100.0,  # Same P&L every period
                inventory=0,
                cash=100.0,
            )
            tracker.snapshots.append(snapshot)

        sharpe = tracker.calculate_sharpe_ratio()
        assert np.isnan(sharpe), "Sharpe should be NaN when variance is zero"

    def test_sample_std_used(self):
        """Verify that sample standard deviation (ddof=1) is used, not population std."""
        tracker = PnLTracker(starting_capital=10000.0, snapshot_interval=60.0)

        # Create snapshots with known P&L values
        base_time = 1000.0
        pnl_values = [0, 100, 150, 200, 300]

        for i, pnl in enumerate(pnl_values):
            snapshot = PnLSnapshot(
                timestamp=Timestamp(base_time + i * 60.0),
                realized_pnl=pnl,
                unrealized_pnl=0.0,
                total_pnl=pnl,
                inventory=0,
                cash=pnl,
            )
            tracker.snapshots.append(snapshot)

        # Calculate expected values
        pnls = np.array(pnl_values)
        equity_changes = np.diff(pnls)
        returns = equity_changes / tracker.starting_capital

        # Sample std (ddof=1) should be used
        sample_std = np.std(returns, ddof=1)
        population_std = np.std(returns, ddof=0)

        # Verify they're different (unless we have very specific data)
        assert sample_std != population_std or len(returns) <= 1, \
            "Sample and population std should differ for this dataset"

        # The tracker should use sample_std
        sharpe = tracker.calculate_sharpe_ratio()
        mean_return = np.mean(returns)
        expected_sharpe_raw = mean_return / sample_std
        periods_per_year = (365.25 * 24 * 3600) / tracker.snapshot_interval
        expected_sharpe = expected_sharpe_raw * np.sqrt(periods_per_year)

        assert np.isclose(sharpe, expected_sharpe, rtol=1e-10), \
            f"Sharpe should use sample std: got {sharpe}, expected {expected_sharpe}"

    def test_annualization_with_different_intervals(self):
        """Test that annualization adjusts correctly for different snapshot intervals."""
        # Test with 60-second intervals
        tracker_60s = PnLTracker(starting_capital=10000.0, snapshot_interval=60.0)

        # Test with 300-second intervals (5 minutes)
        tracker_300s = PnLTracker(starting_capital=10000.0, snapshot_interval=300.0)

        # Create P&L progression with some variance (not constant)
        base_time = 1000.0
        pnl_values = [0, 100, 250, 300, 500, 520]  # Varying increases with some noise

        for i, pnl in enumerate(pnl_values):
            snapshot_60s = PnLSnapshot(
                timestamp=Timestamp(base_time + i * 60.0),
                realized_pnl=pnl,
                unrealized_pnl=0.0,
                total_pnl=pnl,
                inventory=0,
                cash=pnl,
            )
            tracker_60s.snapshots.append(snapshot_60s)

            snapshot_300s = PnLSnapshot(
                timestamp=Timestamp(base_time + i * 300.0),
                realized_pnl=pnl,
                unrealized_pnl=0.0,
                total_pnl=pnl,
                inventory=0,
                cash=pnl,
            )
            tracker_300s.snapshots.append(snapshot_300s)

        sharpe_60s = tracker_60s.calculate_sharpe_ratio()
        sharpe_300s = tracker_300s.calculate_sharpe_ratio()

        # Both should be finite (not inf or nan) with varying returns
        assert np.isfinite(sharpe_60s), "60s Sharpe should be finite with varying returns"
        assert np.isfinite(sharpe_300s), "300s Sharpe should be finite with varying returns"

        # The annualization factor should differ by sqrt(5) since 300s = 5 * 60s
        # More frequent snapshots mean more periods per year, so higher annualized Sharpe
        assert sharpe_60s > sharpe_300s, \
            "60s snapshots should have higher annualized Sharpe than 300s snapshots"

        ratio = sharpe_60s / sharpe_300s
        expected_ratio = np.sqrt(300.0 / 60.0)  # sqrt(5) ≈ 2.236

        assert np.isclose(ratio, expected_ratio, rtol=0.01), \
            f"Annualization ratio should be ~{expected_ratio}, got {ratio}"
