"""
Moving-average feature generators for trend-momentum capture.

* **trailing_moving_average** – standard backward-looking MA  (lagged)
* **centered_moving_average** – symmetric window centred on each point
"""

from __future__ import annotations

from typing import Union

import pandas as pd


def trailing_moving_average(
    series: pd.Series,
    window: int = 7,
    min_periods: int = 1,
) -> pd.Series:
    """
    Trailing (backward-looking) moving average.

    This is appropriate for real-time feature engineering because it only
    uses past observations.

    Parameters
    ----------
    series : pd.Series
        Numeric series.
    window : int
        Rolling window size.
    min_periods : int
        Minimum observations required to produce a value.

    Returns
    -------
    pd.Series
    """
    return (
        series.rolling(window=window, min_periods=min_periods)
        .mean()
        .rename(f"trail_ma_{window}")
    )


def centered_moving_average(
    series: pd.Series,
    window: int = 7,
    min_periods: int = 1,
) -> pd.Series:
    """
    Centered (symmetric) moving average.

    Uses future and past observations — suitable for decomposition /
    offline analysis, **not** for real-time prediction.

    Parameters
    ----------
    series : pd.Series
        Numeric series.
    window : int
        Rolling window size (should be odd for perfect centering).
    min_periods : int
        Minimum observations required.

    Returns
    -------
    pd.Series
    """
    return (
        series.rolling(window=window, center=True, min_periods=min_periods)
        .mean()
        .rename(f"center_ma_{window}")
    )
