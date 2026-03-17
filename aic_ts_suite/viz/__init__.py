"""
Viz sub-package – EDA plots, forecast overlays, and styled display.

Public API:
    plot_seasonal      – seasonal overlay with colour gradients
    plot_acf_pacf      – side-by-side ACF / PACF bar charts
    plot_decomposition – classical 4-panel decomposition (Y = T + S + R)
    plot_forecast      – observed vs. forecast with prediction intervals
"""

from aic_ts_suite.viz.seasonal import plot_seasonal
from aic_ts_suite.viz.acf_pacf import plot_acf_pacf
from aic_ts_suite.viz.decomposition import plot_decomposition
from aic_ts_suite.viz.forecast_plot import plot_forecast
from aic_ts_suite.viz.styles import styled_summary

__all__ = [
    "plot_seasonal",
    "plot_acf_pacf",
    "plot_decomposition",
    "plot_forecast",
    "styled_summary",
]
