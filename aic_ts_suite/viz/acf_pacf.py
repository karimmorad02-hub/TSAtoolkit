"""
ACF / PACF diagnostic plots with 95 % confidence bands.

Generates side-by-side bar charts with ±1.96/√n significance bounds.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 40,
    alpha: float = 0.05,
    title_prefix: str = "",
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """
    Side-by-side ACF and PACF bar charts with confidence bands.

    Parameters
    ----------
    series : pd.Series
        Stationary (or near-stationary) numeric series.
    lags : int
        Maximum lag to compute.
    alpha : float
        Significance level for the confidence band (default 5 %).
    title_prefix : str
        Prepended to each subplot title.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(series.dropna())
    z = 1.96  # for 95 % CI
    conf_bound = z / np.sqrt(n)

    acf_vals = acf(series.dropna(), nlags=lags, fft=True)
    pacf_vals = pacf(series.dropna(), nlags=lags, method="ywm")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, vals, label in zip(
        axes, [acf_vals, pacf_vals], ["ACF", "PACF"]
    ):
        ax.bar(range(len(vals)), vals, width=0.4, color="steelblue", edgecolor="k", linewidth=0.3)
        ax.axhline(conf_bound, linestyle="--", color="red", linewidth=0.8, label=f"95 % CI (±{conf_bound:.3f})")
        ax.axhline(-conf_bound, linestyle="--", color="red", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel(label)
        ax.set_title(f"{title_prefix}{label}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig
