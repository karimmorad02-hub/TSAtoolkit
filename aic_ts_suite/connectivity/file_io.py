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

logger = logging.getLogger(__name__)

# Accepted extensions for directory scanning
_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


# ------------------------------------------------------------------
# CSV
# ------------------------------------------------------------------
def read_csv(
    path: Union[str, Path],
    timestamp_col: str = "timestamp",
    value_col: Optional[str] = None,
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
    value_col : str, optional
        If provided, only keep this value column alongside the timestamp.
    **kwargs
        Forwarded to ``pd.read_csv``.
    """
    df = pd.read_csv(path, parse_dates=[timestamp_col], **kwargs)
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)

    if value_col and value_col in df.columns:
        df = df[["timestamp", value_col]]

    logger.info("CSV loaded: %s (%d rows)", path, len(df))
    return df


# ------------------------------------------------------------------
# Excel
# ------------------------------------------------------------------
def read_excel(
    path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    timestamp_col: str = "timestamp",
    value_col: Optional[str] = None,
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
    value_col : str, optional
        If provided, only keep this value column alongside the timestamp.
    **kwargs
        Forwarded to ``pd.read_excel``.
    """
    df = pd.read_excel(
        path, sheet_name=sheet_name, parse_dates=[timestamp_col], **kwargs
    )
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)

    if value_col and value_col in df.columns:
        df = df[["timestamp", value_col]]

    logger.info("Excel loaded: %s (%d rows)", path, len(df))
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
