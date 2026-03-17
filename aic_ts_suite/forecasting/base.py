"""
Base interface and result container for all forecasters.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import pandas as pd

from aic_ts_suite.config import CONFIG


@dataclass
class ForecastResult:
    """
    Standardised envelope returned by every forecaster.

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    forecast : pd.Series
        Point forecasts (datetime-indexed).
    lower : pd.Series | None
        Lower bound of the prediction interval.
    upper : pd.Series | None
        Upper bound of the prediction interval.
    fitted_values : pd.Series | None
        In-sample fitted values.
    duration_ms : float
        Wall-clock fit + predict time in milliseconds.
    info_criteria : dict
        Information criteria (AICc, AIC, BIC) when available.
    correlation_id : str
        Traceability token propagated from CONFIG.
    extra : dict
        Model-specific metadata (e.g. selected order, λ).
    """

    model_name: str
    forecast: pd.Series
    lower: Optional[pd.Series] = None
    upper: Optional[pd.Series] = None
    fitted_values: Optional[pd.Series] = None
    duration_ms: float = 0.0
    info_criteria: Dict[str, float] = field(default_factory=dict)
    correlation_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseForecaster(ABC):
    """Interface contract for every forecasting module."""

    name: str = "BaseForecaster"

    @abstractmethod
    def fit(
        self, train: Union[pd.Series, pd.DataFrame], **kwargs
    ) -> "BaseForecaster":
        """Fit the model on a training series or DataFrame."""

    @abstractmethod
    def predict(self, horizon: int, **kwargs) -> ForecastResult:
        """Generate forecasts for *horizon* steps ahead."""

    def fit_predict(
        self,
        train: Union[pd.Series, pd.DataFrame],
        horizon: int,
        **kwargs,
    ) -> ForecastResult:
        """Convenience: fit then predict, recording wall-clock time."""
        t0 = time.perf_counter_ns()
        self.fit(train, **kwargs)
        result = self.predict(horizon, **kwargs)
        result.duration_ms = (time.perf_counter_ns() - t0) / 1e6
        result.correlation_id = CONFIG.correlation_id
        return result
