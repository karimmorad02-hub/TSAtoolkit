"""
Univariate forecasters: AutoARIMA, AutoETS, Holt-Winters.

Includes ``auto_select_univariate`` which picks the best model by AICc.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from aic_ts_suite.config import CONFIG
from aic_ts_suite.forecasting.base import BaseForecaster, ForecastResult

logger = logging.getLogger(__name__)


# ======================================================================
# AutoARIMA
# ======================================================================
class AutoARIMAForecaster(BaseForecaster):
    """
    Automatic ARIMA model selection via ``pmdarima.auto_arima``.

    Parameters
    ----------
    seasonal : bool
        Enable seasonal ARIMA (SARIMA).
    m : int
        Seasonal period (ignored when *seasonal=False*).
    """

    name = "AutoARIMA"

    def __init__(self, seasonal: bool = True, m: int = 12, **arima_kwargs) -> None:
        self.seasonal = seasonal
        self.m = m
        self._extra = arima_kwargs
        self._model = None

    def fit(self, train: pd.Series, **kwargs) -> "AutoARIMAForecaster":
        import pmdarima as pm

        self._model = pm.auto_arima(
            train.values,
            seasonal=self.seasonal,
            m=self.m,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            **self._extra,
        )
        logger.info(
            "[%s] AutoARIMA fitted: %s", CONFIG.correlation_id, self._model.order
        )
        return self

    def predict(self, horizon: int, **kwargs) -> ForecastResult:
        alpha = 1 - CONFIG.confidence_level
        fc, ci = self._model.predict(n_periods=horizon, return_conf_int=True, alpha=alpha)

        idx = pd.date_range(
            start=kwargs.get("start"),
            periods=horizon,
            freq=kwargs.get("freq", None),
        ) if kwargs.get("start") else pd.RangeIndex(horizon)

        return ForecastResult(
            model_name=self.name,
            forecast=pd.Series(fc, index=idx, name="forecast"),
            lower=pd.Series(ci[:, 0], index=idx, name="lower"),
            upper=pd.Series(ci[:, 1], index=idx, name="upper"),
            fitted_values=pd.Series(self._model.predict_in_sample(), name="fitted"),
            info_criteria={
                "AICc": self._model.aicc(),
                "AIC": self._model.aic(),
                "BIC": self._model.bic(),
            },
            extra={"order": self._model.order, "seasonal_order": self._model.seasonal_order},
        )


# ======================================================================
# AutoETS
# ======================================================================
class AutoETSForecaster(BaseForecaster):
    """
    Exponential Smoothing State Space model (ETS) – automatic selection.

    Uses ``statsmodels.tsa.exponential_smoothing.ets.ETSModel``.
    """

    name = "AutoETS"

    def __init__(self, seasonal_periods: int = 12) -> None:
        self.seasonal_periods = seasonal_periods
        self._model = None
        self._fit = None

    def fit(self, train: pd.Series, **kwargs) -> "AutoETSForecaster":
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        best_aic = np.inf
        best_fit = None
        best_cfg = None

        # Grid search over a compact ETS space
        errors = ["add", "mul"]
        trends = [None, "add", "mul"]
        seasons = [None, "add", "mul"]

        for e in errors:
            for t in trends:
                for s in seasons:
                    sp = self.seasonal_periods if s else 0
                    try:
                        m = ETSModel(
                            train,
                            error=e,
                            trend=t,
                            seasonal=s,
                            seasonal_periods=sp or None,
                            damped_trend=(t is not None),
                        )
                        res = m.fit(disp=False, maxiter=500)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_fit = res
                            best_cfg = (e, t, s)
                    except Exception:
                        continue

        if best_fit is None:
            raise RuntimeError("No valid ETS configuration found.")

        self._fit = best_fit
        self._model = best_fit.model
        logger.info(
            "[%s] AutoETS selected: error=%s trend=%s seasonal=%s  AIC=%.2f",
            CONFIG.correlation_id,
            *best_cfg,
            best_aic,
        )
        return self

    def predict(self, horizon: int, **kwargs) -> ForecastResult:
        pred = self._fit.get_prediction(
            start=len(self._fit.model.endog),
            end=len(self._fit.model.endog) + horizon - 1,
        )
        summary = pred.summary_frame(alpha=1 - CONFIG.confidence_level)

        return ForecastResult(
            model_name=self.name,
            forecast=summary["mean"],
            lower=summary["pi_lower"],
            upper=summary["pi_upper"],
            fitted_values=pd.Series(self._fit.fittedvalues, name="fitted"),
            info_criteria={
                "AICc": self._fit.aicc,
                "AIC": self._fit.aic,
                "BIC": self._fit.bic,
            },
        )


# ======================================================================
# Holt-Winters
# ======================================================================
class HoltWintersForecaster(BaseForecaster):
    """
    Holt-Winters Exponential Smoothing (additive or multiplicative).
    """

    name = "Holt-Winters"

    def __init__(
        self,
        seasonal: str = "add",
        seasonal_periods: int = 12,
    ) -> None:
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._fit = None

    def fit(self, train: pd.Series, **kwargs) -> "HoltWintersForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        )
        self._fit = model.fit(optimized=True)
        logger.info("[%s] Holt-Winters fitted (seasonal=%s)", CONFIG.correlation_id, self.seasonal)
        return self

    def predict(self, horizon: int, **kwargs) -> ForecastResult:
        fc = self._fit.forecast(horizon)
        # Holt-Winters in statsmodels doesn't produce native PIs;
        # use simulation-based PIs as an approximation.
        try:
            sim = self._fit.simulate(
                horizon, repetitions=500, error="mul"
            )
            lower = sim.quantile(0.025, axis=1)
            upper = sim.quantile(0.975, axis=1)
        except Exception:
            lower = upper = None

        return ForecastResult(
            model_name=self.name,
            forecast=fc,
            lower=lower,
            upper=upper,
            fitted_values=pd.Series(self._fit.fittedvalues, name="fitted"),
            info_criteria={"AICc": self._fit.aicc, "AIC": self._fit.aic, "BIC": self._fit.bic},
        )


# ======================================================================
# Automated model selection (AICc minimisation)
# ======================================================================
def auto_select_univariate(
    train: pd.Series,
    horizon: int,
    seasonal_periods: int = 12,
    **kwargs,
) -> ForecastResult:
    """
    Run AutoARIMA, AutoETS, and Holt-Winters, then select the model
    with the lowest AICc.

    Returns the ``ForecastResult`` of the best model.
    """
    candidates = [
        AutoARIMAForecaster(seasonal=True, m=seasonal_periods),
        AutoETSForecaster(seasonal_periods=seasonal_periods),
        HoltWintersForecaster(seasonal="add", seasonal_periods=seasonal_periods),
    ]

    results: list[ForecastResult] = []
    for model in candidates:
        try:
            res = model.fit_predict(train, horizon, **kwargs)
            results.append(res)
            logger.info(
                "%s → AICc=%.2f  (%.0f ms)",
                res.model_name,
                res.info_criteria.get("AICc", np.inf),
                res.duration_ms,
            )
        except Exception as exc:
            logger.warning("Model %s failed: %s", model.name, exc)

    if not results:
        raise RuntimeError("All univariate models failed.")

    best = min(results, key=lambda r: r.info_criteria.get("AICc", np.inf))
    logger.info("auto_select_univariate → %s", best.model_name)
    return best
