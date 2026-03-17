"""
File I/O helpers – CSV, Excel, and local directory scanning.

All loaders return tidy ``pandas.DataFrame`` objects that are ready to
be passed through the ``DataCleaner`` pipeline.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


def _select_value_cols(
    df: pd.DataFrame,
    timestamp_col_normalised: str,
    value_cols: Optional[Union[str, List[str]]],
) -> pd.DataFrame:
    """Retain only the requested value columns alongside the timestamp."""
    if not value_cols:
        return df
    cols = [value_cols] if isinstance(value_cols, str) else list(value_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Requested value_cols not found in file: {missing}")
    return df[[timestamp_col_normalised] + cols]

logger = logging.getLogger(__name__)

# Accepted extensions for directory scanning
_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


# ------------------------------------------------------------------
# CSV
# ------------------------------------------------------------------
def read_csv(
    path: Union[str, Path],
    timestamp_col: str = "timestamp",
    value_cols: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read a CSV file and return a DataFrame with a parsed datetime index.

    Parameters
    ----------
    path : str | Path
        File path (local or notebook upload).
    timestamp_col : str
        Name of the column containing timestamps.
    value_cols : str | list[str], optional
        If provided, only keep these value columns alongside the timestamp.
        Pass a single string or a list of strings.
    **kwargs
        Forwarded to ``pd.read_csv``.
    """
    df = pd.read_csv(path, parse_dates=[timestamp_col], **kwargs)
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = _select_value_cols(df, "timestamp", value_cols)
    logger.info("CSV loaded: %s (%d rows, %d cols)", path, len(df), len(df.columns))
    return df


# ------------------------------------------------------------------
# Excel
# ------------------------------------------------------------------
def read_excel(
    path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    timestamp_col: str = "timestamp",
    value_cols: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read an Excel workbook (.xlsx / .xls) into a DataFrame.

    Parameters
    ----------
    path : str | Path
        File path.
    sheet_name : str | int
        Sheet to read (name or zero-based index).
    timestamp_col : str
        Name of the column containing timestamps.
    value_cols : str | list[str], optional
        If provided, only keep these value columns alongside the timestamp.
        Pass a single string or a list of strings.
    **kwargs
        Forwarded to ``pd.read_excel``.
    """
    df = pd.read_excel(
        path, sheet_name=sheet_name, parse_dates=[timestamp_col], **kwargs
    )
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = _select_value_cols(df, "timestamp", value_cols)
    logger.info("Excel loaded: %s (%d rows, %d cols)", path, len(df), len(df.columns))
    return df


# ------------------------------------------------------------------
# Directory scanner
# ------------------------------------------------------------------
def scan_directory(
    directory: Union[str, Path],
    recursive: bool = False,
) -> List[Path]:
    """
    Return a sorted list of supported data files in *directory*.

    Parameters
    ----------
    directory : str | Path
        Root directory to scan.
    recursive : bool
        When *True*, scan subdirectories as well.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    pattern = "**/*" if recursive else "*"
    files = sorted(
        p
        for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )
    logger.info("Scanned %s – found %d data files", root, len(files))
    return files
