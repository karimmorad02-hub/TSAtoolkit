"""
TimescaleDB connector for high-frequency sensor data.

Wraps *psycopg2* (or *sqlalchemy*) to execute arbitrary SQL against the
configured TimescaleDB instance and return tidy pandas DataFrames.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from aic_ts_suite.config import CONFIG, TimescaleConfig

logger = logging.getLogger(__name__)


class TimescaleClient:
    """
    Thin query interface to TimescaleDB.

    Parameters
    ----------
    config : TimescaleConfig | None
        Connection parameters.  Falls back to the global ``CONFIG.timescale``.
    """

    def __init__(self, config: Optional[TimescaleConfig] = None) -> None:
        self._cfg = config or CONFIG.timescale
        self._engine = None

    # ------------------------------------------------------------------
    # lazy connection via SQLAlchemy
    # ------------------------------------------------------------------
    def _get_engine(self):
        if self._engine is None:
            try:
                from sqlalchemy import create_engine

                self._engine = create_engine(
                    self._cfg.dsn,
                    pool_pre_ping=True,
                    connect_args={"options": f"-csearch_path={self._cfg.schema}"},
                )
                logger.info("Connected to TimescaleDB at %s", self._cfg.host)
            except Exception as exc:
                logger.error("TimescaleDB connection failed: %s", exc)
                raise
        return self._engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        parse_dates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Execute *sql* and return the result as a DataFrame.

        Parameters
        ----------
        sql : str
            Raw SQL or SQLAlchemy text expression.
        params : dict, optional
            Bind-parameters for parameterised queries.
        parse_dates : list[str], optional
            Columns to coerce to ``datetime64``.
        """
        correlation_id = CONFIG.correlation_id
        logger.info("[%s] Executing query …", correlation_id)
        t0 = time.perf_counter_ns()

        engine = self._get_engine()
        df = pd.read_sql(sql, engine, params=params, parse_dates=parse_dates)

        duration_ms = (time.perf_counter_ns() - t0) / 1e6
        logger.info(
            "[%s] Query returned %d rows in %.1f ms",
            correlation_id,
            len(df),
            duration_ms,
        )
        return df

    def fetch_sensor(
        self,
        table: str,
        sensor_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100_000,
    ) -> pd.DataFrame:
        """
        Convenience method for fetching a single sensor series.

        Parameters
        ----------
        table : str
            Hypertable name.
        sensor_id : str
            Value to filter on the ``sensor_id`` column.
        start, end : str, optional
            ISO-8601 timestamps for the range filter.
        limit : int
            Maximum rows.
        """
        clauses = ["sensor_id = %(sensor_id)s"]
        params: Dict[str, Any] = {"sensor_id": sensor_id}
        if start:
            clauses.append("time >= %(start)s")
            params["start"] = start
        if end:
            clauses.append("time <= %(end)s")
            params["end"] = end

        where = " AND ".join(clauses)
        sql = (
            f"SELECT time, value FROM {table} "  # noqa: S608
            f"WHERE {where} ORDER BY time LIMIT {limit}"
        )
        return self.query(sql, params=params, parse_dates=["time"])

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.info("TimescaleDB connection closed.")
