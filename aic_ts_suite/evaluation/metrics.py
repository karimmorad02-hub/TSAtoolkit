"""
KPI metrics for forecast evaluation.

Implements:
    MAE   – Mean Absolute Error
    RMSE  – Root Mean Square Error
    MAPE  – Mean Absolute Percentage Error  (with near-zero warning)
    sMAPE – Symmetric MAPE
    R²    – Coefficient of determination
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Threshold below which MAPE denominators trigger a warning
_NEAR_ZERO_THRESHOLD = 1e-8


# ======================================================================
# Individual metrics
# ======================================================================
def mae(observed: pd.Series, forecast: pd.Series) -> float:
    """Mean Absolute Error – average error in original units."""
    o, f = _align(observed, forecast)
    return float(np.mean(np.abs(o - f)))


def rmse(observed: pd.Series, forecast: pd.Series) -> float:
    """Root Mean Square Error – penalises large outliers."""
    o, f = _align(observed, forecast)
    return float(np.sqrt(np.mean((o - f) ** 2)))


def mape(observed: pd.Series, forecast: pd.Series) -> float:
    """
    Mean Absolute Percentage Error.

    **Warning**: unreliable when observed values are near zero.
    """
    o, f = _align(observed, forecast)
    if np.any(np.abs(o) < _NEAR_ZERO_THRESHOLD):
        warnings.warn(
            "MAPE: observed values near zero detected – "
            "metric may be misleading. Consider sMAPE instead.",
            RuntimeWarning,
            stacklevel=2,
        )
    denom = np.where(np.abs(o) < _NEAR_ZERO_THRESHOLD, np.nan, o)
    pct = np.abs((o - f) / denom)
    return float(np.nanmean(pct) * 100)


def smape(observed: pd.Series, forecast: pd.Series) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    More robust than MAPE for low-value observations.
    """
    o, f = _align(observed, forecast)
    denom = (np.abs(o) + np.abs(f)) / 2.0
    denom = np.where(denom < _NEAR_ZERO_THRESHOLD, np.nan, denom)
    pct = np.abs(o - f) / denom
    return float(np.nanmean(pct) * 100)


def r_squared(observed: pd.Series, forecast: pd.Series) -> float:
    """
    Coefficient of determination (R²).

    Measures the proportion of variance in the observed series
    explained by the forecast.
    """
    o, f = _align(observed, forecast)
    ss_res = np.sum((o - f) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


# ======================================================================
# Aggregate helper
# ======================================================================
def compute_all_kpis(
    observed: pd.Series,
    forecast: pd.Series,
) -> Dict[str, float]:
    """
    Compute all KPIs at once and return a tidy dictionary.

    Returns
    -------
    dict
        Keys: MAE, RMSE, MAPE, sMAPE, R².
    """
    return {
        "MAE": mae(observed, forecast),
        "RMSE": rmse(observed, forecast),
        "MAPE": mape(observed, forecast),
        "sMAPE": smape(observed, forecast),
        "R²": r_squared(observed, forecast),
    }


# ======================================================================
# Internal helper
# ======================================================================
def _align(
    observed: pd.Series, forecast: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Align two series on their common index and drop NaNs."""
    common = observed.index.intersection(forecast.index)
    if len(common) == 0:
        raise ValueError("Observed and forecast series have no overlapping index.")
    o = observed.loc[common].values.astype(float)
    f = forecast.loc[common].values.astype(float)
    mask = ~(np.isnan(o) | np.isnan(f))
    return o[mask], f[mask]
