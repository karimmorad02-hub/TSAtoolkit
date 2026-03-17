"""
Canonical time-series observation model used across the toolkit.

Every ingested record is normalised into this format so that all
downstream modules (cleaning, viz, forecasting, evaluation) operate
on undeclared-timestamped numeric observations.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimeSeriesObservation:
    """
    Platform-standard observation envelope.

    Parameters
    ----------
    timestamp_ms : int
        Unix epoch **milliseconds**.
    value : float
        Observed numeric measurement.
    series_id : str | None
        Optional identifier for the metric / sensor.
    """

    timestamp_ms: int
    value: float
    series_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def datetime_utc(self) -> dt.datetime:
        """Return the observation time as a timezone-aware UTC datetime."""
        return dt.datetime.fromtimestamp(
            self.timestamp_ms / 1_000, tz=dt.timezone.utc
        )

    @classmethod
    def from_datetime(
        cls,
        timestamp: dt.datetime,
        value: float,
        series_id: Optional[str] = None,
    ) -> "TimeSeriesObservation":
        """Construct from a Python datetime (auto-converted to epoch ms)."""
        epoch_ms = int(timestamp.timestamp() * 1_000)
        return cls(timestamp_ms=epoch_ms, value=value, series_id=series_id)
