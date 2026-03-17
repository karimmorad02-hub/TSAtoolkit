"""
Vector Autoregression (VAR) with integrated Granger causality testing.

Identifies leading / lagging relationships between multiple time-series
and produces multivariate multi-step forecasts with information criteria.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR as _VAR
from statsmodels.tsa.stattools import grangercausalitytests

from aic_ts_suite.config import CONFIG
from aic_ts_suite.forecasting.base import BaseForecaster, ForecastResult

logger = logging.getLogger(__name__)


class VARForecaster(BaseForecaster):
    """
    Vector Autoregression forecaster.

    Parameters
    ----------
    maxlags : int | None
        Maximum lags to evaluate when selecting model order.
    ic : str
        Information criterion for lag selection (``"aic"``, ``"bic"``,
        ``"hqic"``, ``"fpe"``).
    granger_maxlag : int
        Maximum lag for Granger causality tests.
    granger_alpha : float
        Significance level for Granger tests.
    """

    name = "VAR"

    def __init__(
        self,
        maxlags: Optional[int] = 15,
        ic: str = "aic",
        granger_maxlag: int = 12,
        granger_alpha: float = 0.05,
    ) -> None:
        self.maxlags = maxlags
        self.ic = ic
        self.granger_maxlag = granger_maxlag
        self.granger_alpha = granger_alpha
        self._model = None
        self._fit = None
        self._train_df: Optional[pd.DataFrame] = None
        self.granger_results: Dict[str, Dict[str, float]] = {}

    # ---------- Granger helper ----------
    def _run_granger(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Return {(cause→effect): {lag: p-value}} for significant pairs."""
        results = {}
        cols = list(df.columns)
        for target in cols:
            for cause in cols:
                if target == cause:
                    continue
                try:
                    test = grangercausalitytests(
                        df[[target, cause]].dropna(),
                        maxlag=self.granger_maxlag,
                        verbose=False,
                    )
                    min_pval = min(
                        test[lag][0]["ssr_ftest"][1]
                        for lag in test
                    )
                    key = f"{cause} → {target}"
                    results[key] = {"min_p_value": round(min_pval, 6)}
                    if min_pval < self.granger_alpha:
                        logger.info(
                            "[%s] Granger: %s (p=%.4f)",
                            CONFIG.correlation_id,
                            key,
                            min_pval,
                        )
                except Exception:
                    continue
        return results

    # ---------- Fit ----------
    def fit(self, train: pd.Series | pd.DataFrame, **kwargs) -> "VARForecaster":
        """
        Fit the VAR model.

        Parameters
        ----------
        train : pd.DataFrame
            Each column is a separate time-series.
        """
        if isinstance(train, pd.Series):
            raise TypeError("VAR requires a DataFrame with ≥2 columns.")
        self._train_df = train.copy()
        self._model = _VAR(train)
        self._fit = self._model.fit(maxlags=self.maxlags, ic=self.ic)

        # Run Granger causality
        self.granger_results = self._run_granger(train)

        logger.info(
            "[%s] VAR fitted: lag=%d, AIC=%.2f, BIC=%.2f",
            CONFIG.correlation_id,
            self._fit.k_ar,
            self._fit.aic,
            self._fit.bic,
        )
        return self

    # ---------- Predict ----------
    def predict(self, horizon: int, **kwargs) -> ForecastResult:
        alpha = 1 - CONFIG.confidence_level
        fc = self._fit.forecast(
            self._train_df.values[-self._fit.k_ar :], steps=horizon
        )

        # Build forecast index
        last_idx = self._train_df.index[-1]
        if isinstance(last_idx, pd.Timestamp):
            freq = pd.infer_freq(self._train_df.index)
            idx = pd.date_range(start=last_idx, periods=horizon + 1, freq=freq)[1:]
        else:
            idx = pd.RangeIndex(start=len(self._train_df), stop=len(self._train_df) + horizon)

        fc_df = pd.DataFrame(fc, index=idx, columns=self._train_df.columns)

        # Forecast intervals via impulse response bootstrapping
        try:
            irf = self._fit.forecast_interval(
                self._train_df.values[-self._fit.k_ar :],
                steps=horizon,
                alpha=alpha,
            )
            lower_df = pd.DataFrame(irf[1], index=idx, columns=self._train_df.columns)
            upper_df = pd.DataFrame(irf[2], index=idx, columns=self._train_df.columns)
        except Exception:
            lower_df = upper_df = None

        # Return as a ForecastResult (forecast contains the first column)
        first_col = self._train_df.columns[0]
        return ForecastResult(
            model_name=self.name,
            forecast=fc_df[first_col],
            lower=lower_df[first_col] if lower_df is not None else None,
            upper=upper_df[first_col] if upper_df is not None else None,
            info_criteria={
                "AIC": self._fit.aic,
                "BIC": self._fit.bic,
            },
            extra={
                "lag_order": self._fit.k_ar,
                "granger": self.granger_results,
                "full_forecast_df": fc_df,
                "full_lower_df": lower_df,
                "full_upper_df": upper_df,
            },
        )
