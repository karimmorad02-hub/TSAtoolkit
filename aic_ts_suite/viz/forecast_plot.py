"""
Forecast overlay plot: Observed vs. Forecast with 95 % prediction interval.

Optionally renders KPI metric cards as annotated text boxes directly on
the axes so that error statistics are visible alongside the visual
forecast line.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def plot_forecast(
    observed: pd.Series,
    forecast: pd.Series,
    lower: Optional[pd.Series] = None,
    upper: Optional[pd.Series] = None,
    train_end: Optional[pd.Timestamp] = None,
    kpis: Optional[Dict[str, float]] = None,
    title: str = "Forecast vs Observed",
    figsize: tuple = (14, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot historical data alongside forecasted values.

    Parameters
    ----------
    observed : pd.Series
        Full observed series (datetime-indexed).
    forecast : pd.Series
        Model predictions (datetime-indexed, covers the test period).
    lower, upper : pd.Series, optional
        Lower / upper bounds of the 95 % prediction interval.
    train_end : pd.Timestamp, optional
        If given, a vertical dashed line marks the train/test split.
    kpis : dict[str, float], optional
        Dictionary of metric names → values to render as an annotation box.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Existing axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Observed line
    ax.plot(observed.index, observed.values, label="Observed", color="#2c3e50", linewidth=1.0)

    # Forecast line
    ax.plot(forecast.index, forecast.values, label="Forecast", color="#e67e22", linewidth=1.4, linestyle="--")

    # 95 % prediction-interval band
    if lower is not None and upper is not None:
        ax.fill_between(
            lower.index,
            lower.values,
            upper.values,
            color="#e67e22",
            alpha=0.15,
            label="95 % PI",
        )

    # Train/test split marker
    if train_end is not None:
        ax.axvline(train_end, linestyle=":", color="grey", linewidth=0.9, label="Train / Test split")

    # KPI annotation box
    if kpis:
        text_lines = [
            f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
            for k, v in kpis.items()
        ]
        text_str = "\n".join(text_lines)
        props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#bdc3c7", alpha=0.90)
        ax.text(
            0.02,
            0.97,
            text_str,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=props,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
