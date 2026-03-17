"""
Evaluation sub-package – KPI metrics, model comparison, and
AnalyticsEngineClient integration.
"""

from aic_ts_suite.evaluation.metrics import (
    mae,
    rmse,
    mape,
    smape,
    r_squared,
    compute_all_kpis,
)
from aic_ts_suite.evaluation.comparison import ModelComparison
from aic_ts_suite.evaluation.engine_client import AnalyticsEngineClient

__all__ = [
    "mae",
    "rmse",
    "mape",
    "smape",
    "r_squared",
    "compute_all_kpis",
    "ModelComparison",
    "AnalyticsEngineClient",
]
