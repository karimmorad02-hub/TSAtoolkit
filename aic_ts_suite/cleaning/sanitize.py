"""
Sanitisation protocol for handling missing values in time-series data.

Implements the ``sanitize`` function with two configurable strategies:
* ``"interpolate"`` – linear interpolation (default)
* ``"ffill"``       – forward-fill (last observation carried forward)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from aic_ts_suite.config import CONFIG

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = {"interpolate", "ffill"}


def sanitize(
    df: pd.DataFrame,
    strategy: Optional[str] = None,
    value_cols: Optional[list[str]] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Fill missing values in numeric columns using the chosen strategy.

    Parameters
    ----------
    df : DataFrame
        Input data – must contain numeric columns to sanitise.
    strategy : {"interpolate", "ffill"}, optional
        Missing-value strategy.  Defaults to
        ``CONFIG.default_sanitize_strategy``.
    value_cols : list[str], optional
        Columns to sanitise.  If *None*, all numeric columns are treated.
    inplace : bool
        When *True*, modify *df* directly; otherwise return a copy.

    Returns
    -------
    DataFrame
        Sanitised copy (or the same object when *inplace=True*).
    """
    strategy = strategy or CONFIG.default_sanitize_strategy
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown sanitize strategy '{strategy}'. "
            f"Choose from {_VALID_STRATEGIES}."
        )

    out = df if inplace else df.copy()
    cols = value_cols or list(out.select_dtypes("number").columns)

    n_before = out[cols].isna().sum().sum()

    if strategy == "interpolate":
        out[cols] = out[cols].interpolate(method="linear", limit_direction="both")
    elif strategy == "ffill":
        out[cols] = out[cols].ffill()
        # Back-fill any remaining NaNs at the leading edge
        out[cols] = out[cols].bfill()

    n_after = out[cols].isna().sum().sum()
    logger.info(
        "[%s] sanitize(%s): %d NaN → %d NaN",
        CONFIG.correlation_id,
        strategy,
        n_before,
        n_after,
    )
    return out
