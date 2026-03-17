"""
Lag feature generators for supervised ML models (XGBoost, etc.).

Creates lagged copies of the target variable and optional rolling
statistics (mean, std) that capture short-term momentum and volatility.

These features are strictly backward-looking and safe for real-time use.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def lag_features(
    series: pd.Series,
    lags: Union[int, List[int], range] = range(1, 13),
    drop_na: bool = False,
) -> pd.DataFrame:
    """
    Generate lagged copies of a time-series.

    Parameters
    ----------
    series : pd.Series
        Datetime-indexed numeric series.
    lags : int | list[int] | range
        If an *int*, interpreted as ``range(1, lags + 1)``.
        Otherwise an explicit list / range of lag steps to create.
        Each lag *k* produces column ``lag_k`` equal to ``series.shift(k)``.
    drop_na : bool
        When *True*, drop rows that contain any NaN introduced by shifting.

    Returns
    -------
    pd.DataFrame
        DataFrame with the original series in column ``y`` and one
        column per lag (``lag_1``, ``lag_2``, …).

    Examples
    --------
    >>> lags_df = lag_features(ts, lags=6)          # lags 1–6
    >>> lags_df = lag_features(ts, lags=[1, 3, 12]) # explicit list
    >>> lags_df.head()
    """
    if isinstance(lags, int):
        lags = list(range(1, lags + 1))
    else:
        lags = list(lags)
    data = {"y": series}
    for k in lags:
        data[f"lag_{k}"] = series.shift(k)

    df = pd.DataFrame(data, index=series.index)

    if drop_na:
        df = df.dropna()

    logger.info("lag_features: created %d lag columns", len(lags))
    return df


def rolling_lag_features(
    series: pd.Series,
    windows: Union[int, List[int], range] = (3, 6, 12),
    stats: Optional[List[str]] = None,
    drop_na: bool = False,
) -> pd.DataFrame:
    """
    Generate rolling-window statistics as lagged features.

    For each *window* size and *stat* function, a backward-looking rolling
    aggregate is computed.  This captures trend momentum (mean) and
    local volatility (std).

    Parameters
    ----------
    series : pd.Series
        Datetime-indexed numeric series.
    windows : list[int]
        Rolling window sizes.
    stats : list[str], optional
        Aggregation functions to apply.  Defaults to ``["mean", "std"]``.
    drop_na : bool
        Drop rows with any NaN.

    Returns
    -------
    pd.DataFrame
        One column per (window, stat) combination, e.g.
        ``roll_mean_3``, ``roll_std_3``, ``roll_mean_12``, …

    Examples
    --------
    >>> roll_df = rolling_lag_features(ts, windows=[3, 6, 12])
    >>> roll_df.columns.tolist()
    ['roll_mean_3', 'roll_std_3', 'roll_mean_6', 'roll_std_6', ...]
    """
    if stats is None:
        stats = ["mean", "std"]

    if isinstance(windows, int):
        windows = list(range(1, windows + 1))
    else:
        windows = list(windows)
    data: dict[str, pd.Series] = {}

    for w in windows:
        roller = series.rolling(window=w, min_periods=1)
        for stat in stats:
            col_name = f"roll_{stat}_{w}"
            data[col_name] = getattr(roller, stat)()

    df = pd.DataFrame(data, index=series.index)

    if drop_na:
        df = df.dropna()

    logger.info(
        "rolling_lag_features: %d windows × %d stats = %d columns",
        len(windows),
        len(stats),
        len(data),
    )
    return df


def build_supervised_matrix(
    series: pd.Series,
    lags: Union[int, List[int], range] = range(1, 13),
    rolling_windows: Union[int, List[int], range] = (3, 6, 12),
    rolling_stats: Optional[List[str]] = None,
    fourier_k: int = 0,
    fourier_period: float = 12.0,
    seasonal_period: Optional[float] = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Build a complete supervised feature matrix for tree-based models.

    Combines:
    * **Autoregressive lags** (``lag_1`` … ``lag_k``)
    * **Rolling statistics** (``roll_mean_w``, ``roll_std_w``)
    * **Fourier harmonics** (optionally, via ``fourier_K > 0``)

    The target variable is placed in column ``y``.

    Parameters
    ----------
    series : pd.Series
        Datetime-indexed target series.
    lags : list[int] | range
        Lag steps for autoregressive features.
    rolling_windows : list[int]
        Windows for rolling statistics.
    rolling_stats : list[str], optional
        Aggregation functions (default ``["mean", "std"]``).
    fourier_k : int
        Number of Fourier harmonic pairs.  Set to 0 to skip.
    fourier_period : float
        Seasonal period for Fourier terms.
    seasonal_period : float, optional
        Alias for *fourier_period*.  When provided, overrides
        *fourier_period*.
    drop_na : bool
        Remove rows with NaN (necessary before model training).

    Returns
    -------
    pd.DataFrame
        Ready-to-train feature matrix with column ``y`` as the target.
    """
    if seasonal_period is not None:
        fourier_period = seasonal_period

    lag_df = lag_features(series, lags=lags, drop_na=False)
    roll_df = rolling_lag_features(
        series, windows=rolling_windows, stats=rolling_stats, drop_na=False
    )

    parts = [lag_df, roll_df.drop(columns=[], errors="ignore")]

    if fourier_k > 0:
        from aic_ts_suite.features.fourier import fourier_terms

        ft = fourier_terms(n=len(series), period=fourier_period, K=fourier_k)
        ft.index = series.index
        parts.append(ft)

    df = pd.concat(parts, axis=1)
    # Remove duplicate 'y' columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    if drop_na:
        df = df.dropna()

    logger.info(
        "build_supervised_matrix: %d rows × %d features (after drop_na=%s)",
        len(df),
        df.shape[1] - 1,
        drop_na,
    )
    return df
