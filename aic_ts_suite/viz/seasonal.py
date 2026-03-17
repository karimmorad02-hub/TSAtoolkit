"""
Seasonal overlay plot with colour gradients.

Overlays the same periodic window (e.g. month-of-year) across multiple
cycles so that repeating patterns become visually obvious.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


def plot_seasonal(
    series: pd.Series,
    period: int = 12,
    freq_label: str = "Month",
    title: str = "Seasonal Plot",
    cmap: str = "viridis",
    figsize: tuple = (12, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Overlay seasonal cycles with a colour gradient.

    Parameters
    ----------
    series : pd.Series
        Datetime-indexed numeric series.
    period : int
        Number of observations per seasonal cycle (e.g. 12 for monthly).
    freq_label : str
        Label for the x-axis ticks.
    title : str
        Plot title.
    cmap : str
        Matplotlib colourmap name for the gradient.
    figsize : tuple
        Figure size in inches.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    values = series.values
    n_cycles = int(np.ceil(len(values) / period))
    colormap = cm.get_cmap(cmap, n_cycles)

    for i in range(n_cycles):
        start = i * period
        end = min(start + period, len(values))
        segment = values[start:end]
        x = np.arange(len(segment))
        ax.plot(x, segment, color=colormap(i), alpha=0.75, linewidth=1.2)

    # colour bar legend for cycles
    sm = plt.cm.ScalarMappable(
        cmap=colormap, norm=plt.Normalize(vmin=0, vmax=n_cycles - 1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Cycle index")

    ax.set_xlabel(freq_label)
    ax.set_ylabel(series.name or "Value")
    ax.set_title(title)
    fig.tight_layout()
    return fig
