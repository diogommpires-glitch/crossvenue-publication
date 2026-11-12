"""Configuration manager for the public-data, paper-only trading workflow.

Handles loading and validating paper-trading settings. Exchange credentials and
live execution are not part of the supported configuration.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


class ConfigManager:
    """Manage credential-free paper-trading configuration."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager.

        Args:
            config_dir: Path to config directory (relative or absolute).
                       Defaults to '../config' relative to this file.
        """
        if config_dir is None:
            # Default to config directory relative to project root
            project_root = Path(__file__).parent.parent.parent
            self.config_dir = project_root / "config"
        else:
            self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            raise ConfigurationError(f"Config directory not found: {self.config_dir}")

        self.trading_config: Optional[Dict[str, Any]] = None
        self.risk_limits: Optional[Dict[str, Any]] = None
        self._cipher: Optional[Fernet] = None

    def load_config(self, config_file: str = "trading_config.json") -> Dict[str, Any]:
        """Load and validate trading configuration.

        Args:
            config_file: Name of config file in config directory

        Returns:
            Validated configuration dictionary

        Raises:
            ConfigurationError: If config is invalid or missing
        """
        config_path = self.config_dir / config_file

        if not config_path.exists():
            raise ConfigurationError(
                f"Config file not found: {config_path}\n"
                f"Copy trading_config.json.example to trading_config.json and adjust paper settings."
            )

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {config_file}: {e}")

        # Validate required fields
        self._validate_trading_config(config)

        # Reject old credential configs instead of loading or decrypting secrets.
        if config.get("security", {}).get("encrypt_credentials", False):
            raise ConfigurationError("Credential encryption is unsupported in the paper-only project")

        self.trading_config = config
        return config

    def load_risk_limits(self, limits_file: str = "risk_limits.json") -> Dict[str, Any]:
        """Load risk limit configuration.

        Args:
            limits_file: Name of risk limits file in config directory

        Returns:
            Risk limits dictionary

        Raises:
            ConfigurationError: If limits file is invalid or missing
        """
        limits_path = self.config_dir / limits_file

        if not limits_path.exists():
            raise ConfigurationError(f"Risk limits file not found: {limits_path}")

        try:
            with open(limits_path, 'r') as f:
                limits = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {limits_file}: {e}")

        self._validate_risk_limits(limits)
        self.risk_limits = limits
        return limits

    def _validate_trading_config(self, config: Dict[str, Any]) -> None:
        """Validate trading configuration structure.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_sections = ["mode", "exchange", "trading", "strategy", "risk"]
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required section: {section}")

        # Validate mode
        if config["mode"] != "paper":
            raise ConfigurationError("Only paper trading is supported; exchange execution is disabled")

        # The exchange section describes a public market-data source only.
        exchange = config["exchange"]
        if not isinstance(exchange, dict) or not exchange.get("name"):
            raise ConfigurationError("exchange.name is required for public market data")

        # Validate trading section
        trading = config["trading"]
        required_trading = ["symbol", "base_currency", "quote_currency"]
        for field in required_trading:
            if field not in trading or not trading[field]:
                raise ConfigurationError(
                    f"Missing or empty trading field: {field}"
                )

        # Validate strategy section
        strategy = config["strategy"]
        if "type" not in strategy:
            raise ConfigurationError("Strategy type not specified")

        if "risk_aversion" in strategy:
            if not (0.0 <= strategy["risk_aversion"] <= 1.0):
                raise ConfigurationError(
                    "risk_aversion must be between 0.0 and 1.0"
                )

    def _validate_risk_limits(self, limits: Dict[str, Any]) -> None:
        """Validate risk limits structure.

        Args:
            limits: Risk limits dictionary to validate

        Raises:
            ConfigurationError: If limits are invalid
        """
        required_sections = [
            "position_limits",
            "loss_limits",
            "order_limits",
            "market_conditions"
        ]

        for section in required_sections:
            if section not in limits:
                raise ConfigurationError(
                    f"Missing required risk limits section: {section}"
                )

        # Validate numeric limits are positive
        position = limits["position_limits"]
        if position.get("max_position_base", 0) <= 0:
            raise ConfigurationError("max_position_base must be positive")

        loss = limits["loss_limits"]
        if loss.get("max_loss_per_day", 0) <= 0:
            raise ConfigurationError("max_loss_per_day must be positive")

    def get_encryption_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password.

        Args:
            password: User password for encryption
            salt: Optional salt (generated if not provided)

        Returns:
            Encryption key bytes
        """
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt_credentials(
        self,
        config: Dict[str, Any],
        password: str,
        output_file: str = "trading_config.json"
    ) -> None:
        """Reject credential storage; paper trading does not use API secrets."""
        raise NotImplementedError("API credentials are not used in the paper-only project")

    def _decrypt_credentials(self, config: Dict[str, Any]) -> None:
        """Reject legacy credential decryption in the paper-only configuration."""
        raise NotImplementedError("API credentials are not used in the paper-only project")

    def get_mode(self) -> str:
        """Get the only supported trading mode, `paper`.

        Returns:
            Trading mode string

        Raises:
            ConfigurationError: If config not loaded
        """
        if self.trading_config is None:
            raise ConfigurationError("Configuration not loaded")
        return self.trading_config["mode"]

    def is_paper_trading(self) -> bool:
        """Return whether the loaded configuration selects paper mode.

        Returns:
            True for paper mode; invalid real mode configs are rejected on load.
        """
        return self.get_mode() == "paper"

    def get_strategy_params(self) -> Dict[str, Any]:
        """Get strategy parameters from configuration.

        Returns:
            Strategy parameters dictionary
        """
        if self.trading_config is None:
            raise ConfigurationError("Configuration not loaded")
        return self.trading_config["strategy"]

    def get_exchange_config(self) -> Dict[str, Any]:
        """Get public market-data source configuration.

        Returns:
            Exchange configuration dictionary
        """
        if self.trading_config is None:
            raise ConfigurationError("Configuration not loaded")
        return self.trading_config["exchange"]

    def save_config(
        self,
        config: Dict[str, Any],
        filename: str = "trading_config.json"
    ) -> None:
        """Save configuration to file.

        Args:
            config: Configuration dictionary to save
            filename: Output filename in config directory
        """
        output_path = self.config_dir / filename
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

