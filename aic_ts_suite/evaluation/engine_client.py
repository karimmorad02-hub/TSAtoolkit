"""
AnalyticsEngineClient – integration bridge for propagating notebook
results back into the production platform's analytics pipeline.

Every notebook-initiated run propagates a ``correlationId`` (UUID v4) to
ensure R&D results are traceable across system logs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from aic_ts_suite.config import CONFIG, new_correlation_id
from aic_ts_suite.evaluation.metrics import compute_all_kpis
from aic_ts_suite.forecasting.base import ForecastResult

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Immutable record of a single forecasting run."""

    correlation_id: str
    model_name: str
    duration_ms: float
    kpis: Dict[str, float]
    info_criteria: Dict[str, float]
    extra: Dict[str, Any]
    timestamp_epoch_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnalyticsEngineClient:
    """
    Collects run records and provides an export interface compatible
    with the production analytics pipeline.

    Usage::

        client = AnalyticsEngineClient()
        client.log_run(result, observed_test)
        client.summary()
    """

    def __init__(self) -> None:
        self._runs: List[RunRecord] = []

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log_run(
        self,
        result: ForecastResult,
        observed_test: pd.Series,
        correlation_id: Optional[str] = None,
    ) -> RunRecord:
        """
        Compute KPIs and store a traceable run record.

        Parameters
        ----------
        result : ForecastResult
            Output from a forecaster's ``predict`` or ``fit_predict``.
        observed_test : pd.Series
            Ground truth for the test period.
        correlation_id : str, optional
            Override the global correlation ID.
        """
        cid = correlation_id or result.correlation_id or CONFIG.correlation_id
        kpis = compute_all_kpis(observed_test, result.forecast)

        record = RunRecord(
            correlation_id=cid,
            model_name=result.model_name,
            duration_ms=round(result.duration_ms, 2),
            kpis=kpis,
            info_criteria=result.info_criteria,
            extra={k: str(v) for k, v in result.extra.items()},
            timestamp_epoch_ms=int(time.time() * 1_000),
        )
        self._runs.append(record)

        logger.info(
            "[%s] Run logged: %s  MAE=%.4f  RMSE=%.4f  (%.1f ms)",
            cid,
            result.model_name,
            kpis["MAE"],
            kpis["RMSE"],
            result.duration_ms,
        )
        return record

    # ------------------------------------------------------------------
    # Summaries / export
    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Return a tidy DataFrame of all logged runs."""
        rows = []
        for r in self._runs:
            row = {
                "correlationId": r.correlation_id,
                "Model": r.model_name,
                "Duration (ms)": r.duration_ms,
            }
            row.update(r.kpis)
            row.update(r.info_criteria)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_json(self, indent: int = 2) -> str:
        """Serialise all run records to JSON (for pipeline export)."""
        return json.dumps(
            [r.to_dict() for r in self._runs],
            indent=indent,
            default=str,
        )

    def clear(self) -> None:
        """Discard all logged runs."""
        self._runs.clear()
