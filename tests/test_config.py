"""Tests for configuration system.

Tests configuration loading, validation, and risk enforcement.
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ConfigManager, RiskValidator
from config.config_manager import ConfigurationError
from config.risk_validator import RiskViolation, CircuitBreakerTripped


class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_load_example_config_structure(self):
        """Test that example config has valid structure."""
        config_manager = ConfigManager()

        # Load example config (will fail validation due to placeholder values)
        config_path = config_manager.config_dir / "trading_config.json.example"

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Check required sections exist
        assert "mode" in config
        assert "exchange" in config
        assert "trading" in config
        assert "strategy" in config
        assert "risk" in config
        assert "api_key" not in config["exchange"]
        assert "api_secret" not in config["exchange"]

    def test_paper_config_loads_without_credentials(self, tmp_path):
        source = ConfigManager().config_dir / "trading_config.json.example"
        target = tmp_path / "trading_config.json"
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        loaded = ConfigManager(config_dir=str(tmp_path)).load_config()
        assert loaded["mode"] == "paper"
        assert loaded["exchange"]["market_data"] == "public"

    def test_load_risk_limits(self):
        """Test loading risk limits configuration."""
        config_manager = ConfigManager()
        risk_limits = config_manager.load_risk_limits()

        # Check required sections
        assert "position_limits" in risk_limits
        assert "loss_limits" in risk_limits
        assert "order_limits" in risk_limits
        assert "market_conditions" in risk_limits

        # Check specific values
        assert risk_limits["position_limits"]["max_position_base"] > 0
        assert risk_limits["loss_limits"]["max_loss_per_day"] > 0

    def test_validate_mode(self):
        """Test mode validation."""
        config_manager = ConfigManager()

        # Public market data and paper execution need no exchange credentials.
        valid_config = {"mode": "paper", "exchange": {"name": "binance", "market_data": "public"},
                       "trading": {"symbol": "BTCUSDT", "base_currency": "BTC", "quote_currency": "USDT"},
                       "strategy": {"type": "test"}, "risk": {}}

        # Should not raise
        config_manager._validate_trading_config(valid_config)

        # Invalid mode
        invalid_config = valid_config.copy()
        invalid_config["mode"] = "invalid"

        with pytest.raises(ConfigurationError, match="Only paper trading"):
            config_manager._validate_trading_config(invalid_config)

    def test_is_paper_trading(self):
        """Test paper trading mode detection."""
        config_manager = ConfigManager()
        config_manager.trading_config = {"mode": "paper"}

        assert config_manager.is_paper_trading() is True

        config_manager.trading_config = {"mode": "real"}
        assert config_manager.is_paper_trading() is False
        with pytest.raises(ConfigurationError, match="Only paper trading"):
            config_manager._validate_trading_config({
                "mode": "real", "exchange": {"name": "binance"},
                "trading": {"symbol": "BTCUSDT", "base_currency": "BTC", "quote_currency": "USDT"},
                "strategy": {"type": "test"}, "risk": {},
            })


class TestRiskValidator:
    """Test RiskValidator functionality."""

    @pytest.fixture
    def risk_limits(self):
        """Load risk limits for testing."""
        config_manager = ConfigManager()
        return config_manager.load_risk_limits()

    @pytest.fixture
    def validator(self, risk_limits):
        """Create RiskValidator instance."""
        return RiskValidator(risk_limits)

    def test_validate_order_size_limits(self, validator):
        """Test order size validation."""
        # Valid order
        is_valid, error = validator.validate_order("buy", 0.1, 50000.0)
        assert is_valid is True
        assert error is None

        # Too small
        is_valid, error = validator.validate_order("buy", 0.0001, 50000.0)
        assert is_valid is False
        assert "minimum" in error.lower()

        # Too large
        is_valid, error = validator.validate_order("buy", 10.0, 50000.0)
        assert is_valid is False
        assert "maximum" in error.lower()

    def test_position_tracking(self, validator):
        """Test position tracking."""
        initial_base = validator.current_position_base
        initial_quote = validator.current_position_quote

        # Update position
        validator.update_position(base_delta=0.5, quote_delta=-25000.0)

        assert validator.current_position_base == initial_base + 0.5
        assert validator.current_position_quote == initial_quote - 25000.0

    def test_pnl_tracking(self, validator):
        """Test PnL tracking."""
        initial_daily = validator.daily_pnl

        # Record profitable trade
        validator.record_trade_pnl(100.0)
        assert validator.daily_pnl == initial_daily + 100.0
        assert validator.consecutive_losses == 0

        # Record losing trade
        validator.record_trade_pnl(-50.0)
        assert validator.daily_pnl == initial_daily + 50.0
        assert validator.consecutive_losses == 1

    def test_circuit_breaker_large_loss(self, validator):
        """Test circuit breaker on large single loss."""
        max_loss = validator.limits["loss_limits"]["max_loss_per_trade"]

        # Loss exceeding limit should trigger circuit breaker
        with pytest.raises(CircuitBreakerTripped):
            validator.record_trade_pnl(-max_loss - 100)

        assert validator.circuit_breaker_active is True

    def test_rate_limiting(self, validator):
        """Test order rate limiting."""
        max_per_second = validator.limits["order_limits"]["max_orders_per_second"]

        # Record orders up to limit
        for _ in range(max_per_second):
            validator.record_order()

        # Next order should be rate limited
        is_valid, error = validator.validate_order("buy", 0.1, 50000.0)
        assert is_valid is False
        assert "rate limit" in error.lower()

    def test_market_conditions_validation(self, validator):
        """Test market conditions validation."""
        # Normal conditions
        is_valid, error = validator.validate_market_conditions(
            spread_bps=10.0,
            book_depth=50000.0,
            price_change_pct=0.005
        )
        assert is_valid is True

        # Wide spread
        is_valid, error = validator.validate_market_conditions(
            spread_bps=250.0,
            book_depth=50000.0,
            price_change_pct=0.005
        )
        assert is_valid is False
        assert "spread" in error.lower()

        # Thin book
        is_valid, error = validator.validate_market_conditions(
            spread_bps=10.0,
            book_depth=5000.0,
            price_change_pct=0.005
        )
        assert is_valid is False
        assert "depth" in error.lower()

    def test_risk_status(self, validator):
        """Test risk status reporting."""
        status = validator.get_risk_status()

        # Check structure
        assert "circuit_breaker_active" in status
        assert "position" in status
        assert "pnl" in status
        assert "consecutive_losses" in status
        assert "rate_limits" in status

        # Check position details
        assert "base" in status["position"]
        assert "quote" in status["position"]
        assert "utilization_pct" in status["position"]

        # Check PnL details
        assert "daily" in status["pnl"]
        assert "weekly" in status["pnl"]
        assert "monthly" in status["pnl"]


def test_config_integration(tmp_path):
    """Test integration between ConfigManager and RiskValidator."""
    # This would normally use a real config file
    # For testing, we use the existing risk_limits.json

    config_manager = ConfigManager()
    risk_limits = config_manager.load_risk_limits()
    validator = RiskValidator(risk_limits)

    # Validate an order
    is_valid, error = validator.validate_order("buy", 0.05, 50000.0)

    if is_valid:
        validator.record_order()
        validator.update_position(base_delta=0.05, quote_delta=-2500.0)

    # Check status
    status = validator.get_risk_status()
    assert status["position"]["base"] == 0.05
    assert status["position"]["quote"] == -2500.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
