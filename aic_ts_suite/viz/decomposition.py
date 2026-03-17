"""
Classical additive decomposition: Observed = Trend + Seasonal + Residual.

Produces a 4-panel chart plus a reconstruction-error validation metric.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def plot_decomposition(
    series: pd.Series,
    period: int = 12,
    model: str = "additive",
    title: str = "Classical Decomposition",
    figsize: tuple = (13, 9),
) -> plt.Figure:
    """
    4-panel decomposition chart with reconstruction error.

    Parameters
    ----------
    series : pd.Series
        Datetime-indexed numeric series.
    period : int
        Seasonal period (must match the data frequency).
    model : {"additive", "multiplicative"}
        Decomposition model.
    title : str
        Super-title for the figure.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    result = seasonal_decompose(series.dropna(), model=model, period=period)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    components = [
        ("Observed", result.observed),
        ("Trend", result.trend),
        ("Seasonal", result.seasonal),
        ("Residual", result.resid),
    ]

    colours = ["#2c3e50", "#2980b9", "#27ae60", "#e74c3c"]

    for ax, (label, comp), clr in zip(axes, components, colours):
        ax.plot(comp, color=clr, linewidth=0.9)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    # Reconstruction error (additive: Y - T - S - R should → 0)
    if model == "additive":
        reconstructed = result.trend + result.seasonal + result.resid
    else:
        reconstructed = result.trend * result.seasonal * result.resid

    recon_error = (result.observed - reconstructed).dropna()
    mae = np.mean(np.abs(recon_error))
    axes[3].set_xlabel("Time")

    fig.suptitle(f"{title}  (reconstruction MAE = {mae:.6f})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
