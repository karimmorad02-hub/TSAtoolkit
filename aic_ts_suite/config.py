"""
Global configuration for the aic_ts_suite toolkit.

Holds connection strings, default parameters, and traceability helpers.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Optional


def new_correlation_id() -> str:
    """Generate a UUID-v4 correlation ID for run traceability."""
    return str(uuid.uuid4())


@dataclass
class TimescaleConfig:
    """Connection parameters for TimescaleDB.

    Values are resolved from environment variables first, falling back to the
    supplied defaults.  Set the variables below before running in production::

        TSDB_HOST, TSDB_PORT, TSDB_DATABASE, TSDB_USER, TSDB_PASSWORD, TSDB_SCHEMA
    """

    host: str = field(default_factory=lambda: os.environ.get("TSDB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.environ.get("TSDB_PORT", "5432")))
    database: str = field(default_factory=lambda: os.environ.get("TSDB_DATABASE", "tsdb"))
    user: str = field(default_factory=lambda: os.environ.get("TSDB_USER", "readonly"))
    password: str = field(default_factory=lambda: os.environ.get("TSDB_PASSWORD", ""))
    schema: str = field(default_factory=lambda: os.environ.get("TSDB_SCHEMA", "public"))

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:***"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def dsn_with_password(self) -> str:
        """Full DSN including password – avoid logging this value."""
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class ToolkitConfig:
    """Top-level runtime configuration."""

    timescale: TimescaleConfig = field(default_factory=TimescaleConfig)
    correlation_id: str = field(default_factory=new_correlation_id)
    default_sanitize_strategy: str = "interpolate"  # "interpolate" | "ffill"
    confidence_level: float = 0.95
    random_state: int = 42

    def refresh_correlation_id(self) -> str:
        """Rotate the correlation ID and return the new value."""
        self.correlation_id = new_correlation_id()
        return self.correlation_id


# Singleton – importable from anywhere in the toolkit
CONFIG = ToolkitConfig()
