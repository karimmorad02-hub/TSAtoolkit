"""
Forecasting sub-package – modular interface for multiple paradigms.

Univariate   : AutoARIMA, AutoETS, Holt-Winters
Multivariate : VAR (+ Granger causality)
ML / Boosted : XGBoost (lag-based supervised), Prophet
Neural       : NHITS, MLP
Foundation   : TimeGPT (zero-shot)
"""

from aic_ts_suite.forecasting.base import BaseForecaster, ForecastResult
from aic_ts_suite.forecasting.univariate import (
    AutoARIMAForecaster,
    AutoETSForecaster,
    HoltWintersForecaster,
    auto_select_univariate,
)
from aic_ts_suite.forecasting.multivariate import VARForecaster
from aic_ts_suite.forecasting.ml_models import XGBoostForecaster, ProphetForecaster
from aic_ts_suite.forecasting.neural import NHITSForecaster, MLPForecaster, TimeGPTForecaster

__all__ = [
    "BaseForecaster",
    "ForecastResult",
    "AutoARIMAForecaster",
    "AutoETSForecaster",
    "HoltWintersForecaster",
    "auto_select_univariate",
    "VARForecaster",
    "XGBoostForecaster",
    "ProphetForecaster",
    "NHITSForecaster",
    "MLPForecaster",
    "TimeGPTForecaster",
]
