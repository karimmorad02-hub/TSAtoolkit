"""
ModelComparison – side-by-side comparison of multiple ForecastResults.

Generates a leaderboard DataFrame, styled HTML table, and optional
metric-card visualisations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from aic_ts_suite.config import CONFIG
from aic_ts_suite.evaluation.metrics import compute_all_kpis
from aic_ts_suite.forecasting.base import ForecastResult
from aic_ts_suite.viz.styles import styled_summary, metric_cards
from aic_ts_suite.viz.forecast_plot import plot_forecast

logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Collect multiple ``ForecastResult`` objects and compare them on a
    standardised KPI leaderboard.

    Usage::

        cmp = ModelComparison(observed_test)
        cmp.add(result_arima)
        cmp.add(result_nhits)
        cmp.add(result_timegpt)
        cmp.leaderboard()      # → styled HTML table
        cmp.plot_all(observed)  # → overlay plots
    """

    def __init__(self, observed_test: pd.Series) -> None:
        self.observed_test = observed_test
        self._results: List[ForecastResult] = []
        self._kpis: List[Dict] = []

    # ------------------------------------------------------------------
    # Add results
    # ------------------------------------------------------------------
    def add(self, result: ForecastResult) -> "ModelComparison":
        """Register a ForecastResult for comparison."""
        kpis = compute_all_kpis(self.observed_test, result.forecast)
        kpis["Model"] = result.model_name
        kpis["Duration (ms)"] = round(result.duration_ms, 1)
        kpis["correlationId"] = result.correlation_id or CONFIG.correlation_id

        # Include info criteria if available
        for ic_name, ic_val in result.info_criteria.items():
            kpis[ic_name] = round(ic_val, 2)

        self._results.append(result)
        self._kpis.append(kpis)

        logger.info(
            "[%s] Registered %s – MAE=%.4f  RMSE=%.4f  R²=%.4f",
            kpis["correlationId"],
            result.model_name,
            kpis["MAE"],
            kpis["RMSE"],
            kpis["R²"],
        )
        return self

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------
    def leaderboard(self, sort_by: str = "RMSE", show: bool = True) -> pd.DataFrame:
        """
        Return a comparison DataFrame sorted by *sort_by*, and optionally
        render it as a styled HTML table in the notebook.
        """
        df = pd.DataFrame(self._kpis)
        # Re-order columns: Model first, then KPIs, then metadata
        front = ["Model"]
        kpi_cols = ["MAE", "RMSE", "MAPE", "sMAPE", "R²"]
        meta_cols = [c for c in df.columns if c not in front + kpi_cols]
        df = df[front + kpi_cols + meta_cols]

        if sort_by in df.columns:
            df = df.sort_values(sort_by).reset_index(drop=True)

        if show:
            styled_summary(
                df.set_index("Model"),
                caption="Model Comparison Leaderboard",
                show=True,
            )
        return df

    # ------------------------------------------------------------------
    # Visual overlay
    # ------------------------------------------------------------------
    def plot_all(
        self,
        observed_full: pd.Series,
        train_end: Optional[pd.Timestamp] = None,
    ) -> list:
        """
        Generate one train/test forecast plot per model, with KPI
        annotation boxes overlaid.

        Returns a list of matplotlib Figures.
        """
        figs = []
        for res, kpi_dict in zip(self._results, self._kpis):
            kpis_display = {
                k: v
                for k, v in kpi_dict.items()
                if k in {"MAE", "RMSE", "MAPE", "sMAPE", "R²", "Duration (ms)"}
            }
            fig = plot_forecast(
                observed=observed_full,
                forecast=res.forecast,
                lower=res.lower,
                upper=res.upper,
                train_end=train_end,
                kpis=kpis_display,
                title=f"Forecast – {res.model_name}",
            )
            figs.append(fig)
        return figs

    # ------------------------------------------------------------------
    # Metric delta tracking
    # ------------------------------------------------------------------
    def metric_deltas(self, baseline_model: str = "") -> pd.DataFrame:
        """
        Compute metric deltas relative to a baseline model.

        If no baseline is specified, the first registered model is used.
        """
        df = self.leaderboard(show=False)
        if baseline_model:
            base_row = df[df["Model"] == baseline_model]
        else:
            base_row = df.iloc[[0]]

        if base_row.empty:
            raise ValueError(f"Baseline model '{baseline_model}' not found.")

        kpi_cols = ["MAE", "RMSE", "MAPE", "sMAPE", "R²", "Duration (ms)"]
        avail = [c for c in kpi_cols if c in df.columns]
        deltas = df[avail].subtract(base_row[avail].values[0])
        deltas.insert(0, "Model", df["Model"])
        return deltas
