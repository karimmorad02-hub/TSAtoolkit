"""
XGBoost and Prophet forecasters for the aic_ts_suite toolkit.

**XGBoostForecaster** — builds a supervised lag/rolling/Fourier feature
matrix and trains a gradient-boosted tree model.  Prediction intervals
are estimated via quantile regression (``alpha/2`` and ``1 - alpha/2``).

**ProphetForecaster** — wraps Facebook/Meta Prophet for additive /
multiplicative decomposition with built-in uncertainty intervals,
holiday effects, and custom regressors.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from aic_ts_suite.config import CONFIG
from aic_ts_suite.forecasting.base import BaseForecaster, ForecastResult

logger = logging.getLogger(__name__)


# ======================================================================
# XGBoost Forecaster
# ======================================================================
class XGBoostForecaster(BaseForecaster):
    """
    Gradient-boosted tree forecaster backed by XGBoost.

    Internally builds a supervised feature matrix using autoregressive
    lags, rolling statistics, and optional Fourier harmonics, then fits
    an ``xgboost.XGBRegressor``.

    Prediction intervals are produced by fitting two additional quantile-
    regression models at the ``alpha/2`` and ``1 - alpha/2`` levels.

    Parameters
    ----------
    lags : int | list[int] | range
        Autoregressive lag steps.  An *int* is interpreted as
        ``range(1, lags + 1)``.
    rolling_windows : int | list[int]
        Rolling window sizes for mean/std features.
    fourier_k : int
        Number of Fourier harmonic pairs (0 = disabled).
    fourier_period : float
        Period for Fourier features.
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Shrinkage factor.
    max_depth : int
        Maximum tree depth.
    xgb_params : dict, optional
        Extra keyword arguments forwarded to ``XGBRegressor``.
    """

    name = "XGBoost"

    def __init__(
        self,
        lags: Union[int, List[int], range] = range(1, 13),
        rolling_windows: Union[int, List[int], range] = (3, 6, 12),
        fourier_k: int = 0,
        fourier_period: float = 12.0,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        xgb_params: Optional[Dict] = None,
    ) -> None:
        self.lags = list(range(1, lags + 1)) if isinstance(lags, int) else list(lags)
        self.rolling_windows = (
            list(range(1, rolling_windows + 1))
            if isinstance(rolling_windows, int)
            else list(rolling_windows)
        )
        self.fourier_k = fourier_k
        self.fourier_period = fourier_period
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self._xgb_params = xgb_params or {}

        self._model = None
        self._model_lo = None
        self._model_hi = None
        self._train_series: Optional[pd.Series] = None
        self._feature_cols: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Internal: build features
    # ------------------------------------------------------------------
    def _build_features(self, series: pd.Series) -> pd.DataFrame:
        from aic_ts_suite.features.lags import build_supervised_matrix

        return build_supervised_matrix(
            series,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            fourier_k=self.fourier_k,
            fourier_period=self.fourier_period,
            drop_na=True,
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, train: pd.Series, **kwargs) -> "XGBoostForecaster":
        """
        Fit XGBoost on a supervised lag matrix built from *train*.

        Also fits two quantile models for prediction intervals.
        """
        from xgboost import XGBRegressor

        self._train_series = train.copy()
        mat = self._build_features(train)

        y = mat["y"].values
        X = mat.drop(columns=["y"]).values
        self._feature_cols = [c for c in mat.columns if c != "y"]

        # Point model
        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=CONFIG.random_state,
            verbosity=0,
            **self._xgb_params,
        )
        self._model.fit(X, y)

        # Quantile models for PI
        alpha = 1 - CONFIG.confidence_level
        self._model_lo = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="reg:quantileerror",
            quantile_alpha=alpha / 2,
            random_state=CONFIG.random_state,
            verbosity=0,
        )
        self._model_lo.fit(X, y)

        self._model_hi = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="reg:quantileerror",
            quantile_alpha=1 - alpha / 2,
            random_state=CONFIG.random_state,
            verbosity=0,
        )
        self._model_hi.fit(X, y)

        logger.info(
            "[%s] XGBoost fitted on %d rows × %d features",
            CONFIG.correlation_id,
            len(y),
            X.shape[1],
        )
        return self

    # ------------------------------------------------------------------
    # Predict (recursive multi-step)
    # ------------------------------------------------------------------
    def predict(self, horizon: int, **kwargs) -> ForecastResult:
        """
        Generate *horizon* step-ahead forecasts using recursive
        (iterative) prediction.

        At each step the latest prediction is appended to the history
        so that subsequent lag features can be computed.
        """
        history = self._train_series.copy()
        preds, preds_lo, preds_hi = [], [], []

        freq = pd.infer_freq(history.index) or "MS"

        for _ in range(horizon):
            mat = self._build_features(history)
            X_last = mat.drop(columns=["y"]).iloc[[-1]].values
            yhat = float(self._model.predict(X_last)[0])
            yhat_lo = float(self._model_lo.predict(X_last)[0])
            yhat_hi = float(self._model_hi.predict(X_last)[0])

            next_ts = history.index[-1] + pd.tseries.frequencies.to_offset(freq)
            history = pd.concat([history, pd.Series([yhat], index=[next_ts], name=history.name)])

            preds.append(yhat)
            preds_lo.append(yhat_lo)
            preds_hi.append(yhat_hi)

        idx = history.index[-horizon:]

        # In-sample fitted values
        mat_full = self._build_features(self._train_series)
        X_full = mat_full.drop(columns=["y"]).values
        fitted = pd.Series(
            self._model.predict(X_full),
            index=mat_full.index,
            name="fitted",
        )

        return ForecastResult(
            model_name=self.name,
            forecast=pd.Series(preds, index=idx, name="forecast"),
            lower=pd.Series(preds_lo, index=idx, name="lower"),
            upper=pd.Series(preds_hi, index=idx, name="upper"),
            fitted_values=fitted,
            extra={
                "lags": self.lags,
                "rolling_windows": self.rolling_windows,
                "fourier_k": self.fourier_k,
                "n_features": len(self._feature_cols),
            },
        )

    # ------------------------------------------------------------------
    # Convenience: feature importance
    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.Series:
        """
        Return feature importances (gain-based) as a sorted Series.
        """
        if self._model is None:
            raise RuntimeError("Call .fit() first.")
        imp = pd.Series(
            self._model.feature_importances_,
            index=self._feature_cols,
            name="importance",
        ).sort_values(ascending=False)
        return imp


# ======================================================================
# Prophet Forecaster
# ======================================================================
class ProphetForecaster(BaseForecaster):
    """
    Facebook/Meta Prophet forecaster.

    Supports additive and multiplicative seasonality, holiday effects,
    and custom regressors.  Prediction intervals are natively provided
    by Prophet's posterior sampling.

    Parameters
    ----------
    growth : {"linear", "logistic", "flat"}
        Trend growth model.
    seasonality_mode : {"additive", "multiplicative"}
        How seasonal components combine with the trend.
    yearly_seasonality : bool | int
        Enable yearly seasonality (or set Fourier order).
    weekly_seasonality : bool | int
        Enable weekly seasonality.
    daily_seasonality : bool | int
        Enable daily seasonality.
    changepoint_prior_scale : float
        Flexibility of the trend changepoints.
    prophet_kwargs : dict, optional
        Additional keyword arguments forwarded to ``Prophet()``.
    """

    name = "Prophet"

    def __init__(
        self,
        growth: str = "linear",
        seasonality_mode: str = "additive",
        yearly_seasonality: Union[bool, int] = True,
        weekly_seasonality: Union[bool, int] = False,
        daily_seasonality: Union[bool, int] = False,
        changepoint_prior_scale: float = 0.05,
        prophet_kwargs: Optional[Dict] = None,
    ) -> None:
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self._prophet_kwargs = prophet_kwargs or {}

        self._model = None
        self._train_df: Optional[pd.DataFrame] = None
        self._freq: Optional[str] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_prophet_df(series: pd.Series) -> pd.DataFrame:
        """Convert a datetime-indexed Series to Prophet's expected format."""
        idx = series.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        return pd.DataFrame({"ds": idx, "y": series.values})

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, train: pd.Series, **kwargs) -> "ProphetForecaster":
        """
        Fit a Prophet model.

        Parameters
        ----------
        train : pd.Series
            Datetime-indexed training series.
        **kwargs
            ``holidays`` (pd.DataFrame) and ``extra_regressors``
            (list[str]) can be passed here.
        """
        from prophet import Prophet

        self._freq = pd.infer_freq(train.index) or "MS"
        self._train_df = self._to_prophet_df(train)

        self._model = Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            interval_width=CONFIG.confidence_level,
            **self._prophet_kwargs,
        )

        # Optional: add custom holidays
        holidays = kwargs.get("holidays")
        if holidays is not None:
            self._model.holidays = holidays

        self._model.fit(self._train_df)

        logger.info(
            "[%s] Prophet fitted (growth=%s, seasonality=%s)",
            CONFIG.correlation_id,
            self.growth,
            self.seasonality_mode,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, horizon: int, **kwargs) -> ForecastResult:
        """
        Produce *horizon* step-ahead forecasts with prediction intervals.
        """
        future = self._model.make_future_dataframe(
            periods=horizon, freq=self._freq
        )
        prediction = self._model.predict(future)

        # Separate forecast (out-of-sample) from fitted (in-sample)
        n_train = len(self._train_df)
        fc_rows = prediction.iloc[n_train:]
        in_sample = prediction.iloc[:n_train]

        idx = pd.DatetimeIndex(fc_rows["ds"])

        return ForecastResult(
            model_name=self.name,
            forecast=pd.Series(fc_rows["yhat"].values, index=idx, name="forecast"),
            lower=pd.Series(fc_rows["yhat_lower"].values, index=idx, name="lower"),
            upper=pd.Series(fc_rows["yhat_upper"].values, index=idx, name="upper"),
            fitted_values=pd.Series(
                in_sample["yhat"].values,
                index=pd.DatetimeIndex(in_sample["ds"]),
                name="fitted",
            ),
            extra={
                "growth": self.growth,
                "seasonality_mode": self.seasonality_mode,
                "changepoints": list(self._model.changepoints.astype(str)),
            },
        )
