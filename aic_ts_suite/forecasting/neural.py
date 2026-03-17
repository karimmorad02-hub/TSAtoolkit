"""
Neural and foundation-model forecasters: NHITS, MLP, TimeGPT.

* NHITS and MLP are powered by **neuralforecast** (Nixtla).
* TimeGPT uses the **nixtla** SDK for zero-shot inference.

If a dependency is missing, the forecaster raises a clear ImportError
at construction time rather than at call time.
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
# Helper: convert to NeuralForecast long-format
# ======================================================================
def _to_nf_df(series: pd.Series, uid: str = "series_1") -> pd.DataFrame:
    """Convert a datetime-indexed series to NeuralForecast long format."""
    df = pd.DataFrame({"ds": series.index, "y": series.values})
    df["unique_id"] = uid
    return df[["unique_id", "ds", "y"]]


# ======================================================================
# NHITS
# ======================================================================
class NHITSForecaster(BaseForecaster):
    """
    N-HiTS (Neural Hierarchical Interpolation for Time Series).

    Parameters
    ----------
    horizon : int
        Forecast horizon (also used for training window configuration).
    max_steps : int
        Training epochs.
    """

    name = "NHITS"

    def __init__(self, horizon: int = 12, max_steps: int = 200) -> None:
        self.horizon = horizon
        self.max_steps = max_steps
        self._nf = None
        self._train_df = None

    def fit(self, train: pd.Series, **kwargs) -> "NHITSForecaster":
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS

        self._train_df = _to_nf_df(train)
        model = NHITS(
            h=self.horizon,
            input_size=2 * self.horizon,
            max_steps=self.max_steps,
            scaler_type="standard",
        )
        self._nf = NeuralForecast(models=[model], freq=pd.infer_freq(train.index) or "MS")
        self._nf.fit(df=self._train_df)
        logger.info("[%s] NHITS fitted (%d steps)", CONFIG.correlation_id, self.max_steps)
        return self

    def predict(self, horizon: int, **kwargs) -> ForecastResult:
        preds = self._nf.predict(df=self._train_df)
        fc = preds["NHITS"].values[:horizon]
        idx = preds["ds"].values[:horizon]
        return ForecastResult(
            model_name=self.name,
            forecast=pd.Series(fc, index=idx, name="forecast"),
        )


# ======================================================================
# MLP
# ======================================================================
class MLPForecaster(BaseForecaster):
    """
    Simple feed-forward MLP via NeuralForecast.
    """

    name = "MLP"

    def __init__(self, horizon: int = 12, max_steps: int = 200) -> None:
        self.horizon = horizon
        self.max_steps = max_steps
        self._nf = None
        self._train_df = None

    def fit(self, train: pd.Series, **kwargs) -> "MLPForecaster":
        from neuralforecast import NeuralForecast
        from neuralforecast.models import MLP

        self._train_df = _to_nf_df(train)
        model = MLP(
            h=self.horizon,
            input_size=2 * self.horizon,
            max_steps=self.max_steps,
            scaler_type="standard",
        )
        self._nf = NeuralForecast(models=[model], freq=pd.infer_freq(train.index) or "MS")
        self._nf.fit(df=self._train_df)
        logger.info("[%s] MLP fitted (%d steps)", CONFIG.correlation_id, self.max_steps)
        return self

    def predict(self, horizon: int, **kwargs) -> ForecastResult:
        preds = self._nf.predict(df=self._train_df)
        fc = preds["MLP"].values[:horizon]
        idx = preds["ds"].values[:horizon]
        return ForecastResult(
            model_name=self.name,
            forecast=pd.Series(fc, index=idx, name="forecast"),
        )


# ======================================================================
# TimeGPT (zero-shot foundation model)
# ======================================================================
class TimeGPTForecaster(BaseForecaster):
    """
    Zero-shot forecasting using Nixtla's TimeGPT API.

    Requires a ``NIXTLA_API_KEY`` environment variable or an explicit
    ``api_key`` parameter.
    """

    name = "TimeGPT"

    def __init__(self, api_key: Optional[str] = None, horizon: int = 12) -> None:
        self.api_key = api_key
        self.horizon = horizon
        self._client = None
        self._train_df = None

    def fit(self, train: pd.Series, **kwargs) -> "TimeGPTForecaster":
        import os
        from nixtla import NixtlaClient

        key = self.api_key or os.environ.get("NIXTLA_API_KEY", "")
        self._client = NixtlaClient(api_key=key)
        self._train_df = _to_nf_df(train)
        logger.info("[%s] TimeGPT client initialised (zero-shot)", CONFIG.correlation_id)
        return self

    def predict(self, horizon: int, **kwargs) -> ForecastResult:
        freq = kwargs.get("freq", "MS")
        preds = self._client.forecast(
            df=self._train_df,
            h=horizon,
            freq=freq,
            level=[int(CONFIG.confidence_level * 100)],
        )

        fc = preds["TimeGPT"].values
        idx = preds["ds"].values
        ci_lo = preds.get(f"TimeGPT-lo-{int(CONFIG.confidence_level*100)}")
        ci_hi = preds.get(f"TimeGPT-hi-{int(CONFIG.confidence_level*100)}")

        return ForecastResult(
            model_name=self.name,
            forecast=pd.Series(fc, index=idx, name="forecast"),
            lower=pd.Series(ci_lo.values, index=idx, name="lower") if ci_lo is not None else None,
            upper=pd.Series(ci_hi.values, index=idx, name="upper") if ci_hi is not None else None,
        )
