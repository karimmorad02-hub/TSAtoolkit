"""Features sub-package – Fourier harmonics, moving averages & lag features."""

from aic_ts_suite.features.fourier import fourier_terms, optimal_k
from aic_ts_suite.features.moving_averages import (
    trailing_moving_average,
    centered_moving_average,
)
from aic_ts_suite.features.lags import (
    lag_features,
    rolling_lag_features,
    build_supervised_matrix,
)

__all__ = [
    "fourier_terms",
    "optimal_k",
    "trailing_moving_average",
    "centered_moving_average",
    "lag_features",
    "rolling_lag_features",
    "build_supervised_matrix",
]
