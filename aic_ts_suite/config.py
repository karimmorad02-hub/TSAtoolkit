"""
Global configuration for the aic_ts_suite toolkit.

Holds connection strings, default parameters, and traceability helpers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


def new_correlation_id() -> str:
    """Generate a UUID-v4 correlation ID for run traceability."""
    return str(uuid.uuid4())


@dataclass
class TimescaleConfig:
    """Connection parameters for TimescaleDB."""

    host: str = "10.20.0.10"
    port: int = 5432
    database: str = "tsdb"
    user: str = "readonly"
    password: str = ""
    schema: str = "public"

    @property
    def dsn(self) -> str:
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
