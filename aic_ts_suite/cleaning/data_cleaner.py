"""
DataCleaner – high-level pipeline for loading, sanitising, and
normalising time-series DataFrames from CSV / Excel sources.

Typical usage::

    cleaner = DataCleaner("data/raw/sensor.csv", timestamp_col="date")
    df = cleaner.load().sanitize().to_epoch_ms().result()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from aic_ts_suite.cleaning.sanitize import sanitize
from aic_ts_suite.connectivity.file_io import read_csv, read_excel

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Fluent builder for data ingestion + cleaning.

    Parameters
    ----------
    source : str | Path
        Path to a CSV or Excel file.
    timestamp_col : str
        Name of the raw timestamp column.
    value_cols : str | list[str] | None
        Optional column(s) to retain alongside the timestamp.  When *None*
        all columns are kept.
    sheet_name : str | int
        Excel sheet (ignored for CSV).
    """

    def __init__(
        self,
        source: Union[str, Path],
        timestamp_col: str = "timestamp",
        value_cols: Optional[Union[str, List[str]]] = None,
        sheet_name: Union[str, int] = 0,
    ) -> None:
        self._source = Path(source)
        self._ts_col = timestamp_col
        self._val_cols = value_cols
        self._sheet = sheet_name
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Pipeline steps (fluent)
    # ------------------------------------------------------------------
    def load(self) -> "DataCleaner":
        """Read the file into an internal DataFrame."""
        ext = self._source.suffix.lower()
        if ext == ".csv":
            self._df = read_csv(
                self._source,
                timestamp_col=self._ts_col,
                value_cols=self._val_cols,
            )
        elif ext in {".xlsx", ".xls"}:
            self._df = read_excel(
                self._source,
                sheet_name=self._sheet,
                timestamp_col=self._ts_col,
                value_cols=self._val_cols,
            )
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        return self

    def sanitize(self, strategy: Optional[str] = None) -> "DataCleaner":
        """Apply the sanitise protocol to fill missing values."""
        self._ensure_loaded()
        self._df = sanitize(self._df, strategy=strategy)
        return self

    def to_epoch_ms(self, col: str = "timestamp") -> "DataCleaner":
        """
        Convert a datetime column to Unix epoch milliseconds (int64).

        This aligns the data with the platform's ``TimeSeriesObservation``
        format.
        """
        self._ensure_loaded()
        if col in self._df.columns:
            ts = pd.to_datetime(self._df[col])
            self._df["timestamp_ms"] = (
                ts.astype(np.int64) // 10**6
            )
        return self

    def set_datetime_index(self, col: str = "timestamp") -> "DataCleaner":
        """Set the datetime column as the DataFrame index."""
        self._ensure_loaded()
        if col in self._df.columns:
            self._df[col] = pd.to_datetime(self._df[col])
            self._df = self._df.set_index(col)
        return self

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------
    def result(self) -> pd.DataFrame:
        """Return the current state of the cleaned DataFrame."""
        self._ensure_loaded()
        return self._df.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._df is None:
            raise RuntimeError(
                "No data loaded yet. Call .load() before other pipeline steps."
            )

    def __repr__(self) -> str:  # pragma: no-cover
        rows = len(self._df) if self._df is not None else 0
        cols = list(self._df.columns) if self._df is not None else []
        return f"<DataCleaner source={self._source.name!r} rows={rows} cols={cols}>"
