"""
Fourier harmonic term generation for modelling complex or non-integer
seasonal periods (e.g. 52.18 weeks/year).

Provides:
* ``fourier_terms`` – compute sin/cos pairs for a given K
* ``optimal_k``     – select K that minimises AICc on a linear model
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fourier_terms(
    n: int,
    period: float,
    K: int,
) -> pd.DataFrame:
    """
    Generate K pairs of sin/cos Fourier harmonic features.

    Parameters
    ----------
    n : int
        Number of observations (rows).
    period : float
        Seasonal period length (can be non-integer, e.g. 52.18).
    K : int
        Number of harmonic pairs (1 ≤ K ≤ floor(period / 2)).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``sin_1, cos_1, …, sin_K, cos_K``.
    """
    max_k = int(np.floor(period / 2))
    if K > max_k:
        raise ValueError(
            f"K={K} exceeds max harmonics for period={period} (max K={max_k})"
        )

    t = np.arange(1, n + 1)
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)

    return pd.DataFrame(cols)


def _aicc(rss: float, n: int, p: int) -> float:
    """Compute corrected AIC (AICc) from residual sum of squares."""
    if n - p - 1 <= 0:
        return np.inf
    aic = n * np.log(rss / n) + 2 * p
    correction = (2 * p * (p + 1)) / (n - p - 1)
    return aic + correction


def optimal_k(
    y: pd.Series,
    period: float,
    max_K: Optional[int] = None,
) -> int:
    """
    Select the optimal number of Fourier harmonics K using AICc.

    Fits a simple OLS regression of *y* on Fourier features for each
    candidate K and returns the one that minimises AICc.

    Parameters
    ----------
    y : pd.Series
        Time-series values.
    period : float
        Seasonal period.
    max_K : int, optional
        Upper bound on K.  Defaults to ``floor(period / 2)``.

    Returns
    -------
    int
        Selected number of harmonics.
    """
    n = len(y)
    y_arr = y.values.astype(float)
    if max_K is None:
        max_K = int(np.floor(period / 2))

    best_k = 1
    best_aicc = np.inf

    for k in range(1, max_K + 1):
        X = fourier_terms(n, period, k).values
        # Add intercept
        X_aug = np.column_stack([np.ones(n), X])
        # OLS via normal equations
        try:
            beta = np.linalg.lstsq(X_aug, y_arr, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        residuals = y_arr - X_aug @ beta
        rss = np.sum(residuals**2)
        p = X_aug.shape[1]  # number of parameters
        score = _aicc(rss, n, p)
        if score < best_aicc:
            best_aicc = score
            best_k = k

    logger.info(
        "optimal_k: selected K=%d (AICc=%.2f) for period=%.2f",
        best_k,
        best_aicc,
        period,
    )
    return best_k
