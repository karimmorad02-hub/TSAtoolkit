#!/usr/bin/env python3
"""
toolkit_demo.py — Full end-to-end demonstration of the aic_ts_suite toolkit.

Walks through every module in sequence:
    1.  Configuration & Traceability     (config)
    2.  Data Ingestion (CSV)             (connectivity)
    3.  Data Cleaning & Sanitisation     (cleaning)
    4.  Variance-Stabilising Transforms  (signals.transforms)
    5.  EDA Visualisation                (viz)
    6.  Feature Engineering              (features)
    7.  Univariate Forecasting           (forecasting.univariate)
    7b. XGBoost Forecasting              (forecasting.ml_models)
    7c. Prophet Forecasting              (forecasting.ml_models)
    8.  Multivariate Forecasting (VAR)   (forecasting.multivariate)
    9.  KPI Evaluation & Comparison      (evaluation)
    10. Styled Display & Metric Cards    (display)
    11. Run Traceability                 (evaluation.engine_client)

Usage
-----
    python scripts/toolkit_demo.py [OPTIONS]

Options
-------
    --csv-path       Path to the input CSV file (required).
    --timestamp-col  Column name containing timestamps (default: utc_timestamp).
    --value-cols     Space-separated list of feature columns to load.
    --target-col     Primary target column for univariate forecasting
                     (defaults to the first value column).
    --horizon        Forecast horizon in steps (default: 30).
    --seasonal       Dominant seasonal period (default: 12).
    --freq           Pandas DateOffset alias for data frequency (default: D).
    --verbose, -v    Enable verbose progress output.
    --no-plots       Skip all matplotlib figures (useful for headless runs).

Examples
--------
    # Minimal run — uses sensible defaults:
    python scripts/toolkit_demo.py --csv-path Data/time_series_15min_singleindex.csv

    # Verbose run with custom settings:
    python scripts/toolkit_demo.py \\
        --csv-path Data/time_series_15min_singleindex.csv \\
        --value-cols AT_load_actual_entsoe_transparency AT_solar_generation_actual AT_wind_onshore_generation_actual \\
        --horizon 30 --seasonal 12 --freq D --verbose
"""

import argparse
import logging
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ensure the toolkit package is importable when run from any working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


def _section(title: str) -> None:
    """Print a prominent section banner when verbose logging is enabled."""
    log.info("")
    log.info("=" * 70)
    log.info("  %s", title)
    log.info("=" * 70)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="toolkit_demo",
        description="End-to-end aic_ts_suite demonstration script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Data ---
    parser.add_argument(
        "--csv-path",
        default="/home/user14502/TSAtoolkit/Data/time_series_15min_singleindex.csv",
        help="Path to input CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--timestamp-col",
        default="utc_timestamp",
        help="Timestamp column name (default: %(default)s)",
    )
    parser.add_argument(
        "--value-cols",
        nargs="+",
        default=[
            "AT_load_actual_entsoe_transparency",
            "AT_solar_generation_actual",
            "AT_wind_onshore_generation_actual",
        ],
        metavar="COL",
        help="Feature columns to load (default: %(default)s)",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Primary univariate target column (default: first value-col)",
    )

    # --- Modelling ---
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Forecast horizon in steps (default: %(default)s)",
    )
    parser.add_argument(
        "--seasonal",
        type=int,
        default=12,
        dest="seasonal_period",
        help="Dominant seasonal period (default: %(default)s)",
    )
    parser.add_argument(
        "--freq",
        default="D",
        help="Pandas DateOffset alias for data frequency (default: %(default)s)",
    )

    # --- Behaviour ---
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose progress output",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all matplotlib figures (headless / CI mode)",
    )

    args = parser.parse_args(argv)
    if args.target_col is None:
        args.target_col = args.value_cols[0]
    return args


# ---------------------------------------------------------------------------
# Section 1 — Configuration & Traceability
# ---------------------------------------------------------------------------

def section_config(verbose: bool) -> object:
    """Load global CONFIG and log correlation ID and key settings."""
    _section("1 — Configuration & Traceability")
    from aic_ts_suite.config import CONFIG  # noqa: PLC0415

    log.info("Correlation ID : %s", CONFIG.correlation_id)
    log.info("TimescaleDB DSN: %s", CONFIG.timescale.dsn)
    log.info("Confidence     : %d %%", CONFIG.confidence_level * 100)
    log.info("Sanitise strat : %s", CONFIG.default_sanitize_strategy)
    return CONFIG


# ---------------------------------------------------------------------------
# Section 2 — Data Ingestion (CSV)
# ---------------------------------------------------------------------------

def section_ingest(csv_path: str, timestamp_col: str, value_cols: list, target_col: str) -> pd.DataFrame:
    """
    Load a CSV time-series file and return a cleaned, sorted DataFrame.

    Parameters
    ----------
    csv_path : str
        Absolute or relative path to the CSV file.
    timestamp_col : str
        Name of the column containing ISO-formatted timestamps.
    value_cols : list of str
        Feature columns to retain.
    target_col : str
        Primary series used in univariate forecasting steps.

    Returns
    -------
    pd.DataFrame
        Columns: ['timestamp'] + value_cols, sorted by timestamp.
    """
    _section("2 — Data Ingestion (CSV)")
    from aic_ts_suite.connectivity.models import TimeSeriesObservation  # noqa: PLC0415

    raw_df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
    raw_df = raw_df[[timestamp_col] + value_cols].rename(columns={timestamp_col: "timestamp"})
    raw_df = raw_df.sort_values("timestamp").reset_index(drop=True)

    log.info("Raw shape : %s  |  Features: %s", raw_df.shape, value_cols)
    log.info("NaN counts:\n%s", raw_df[value_cols].isna().sum().to_string())

    # Log the first observation in the platform-standard envelope
    first_row = raw_df.iloc[0]
    obs = TimeSeriesObservation.from_datetime(
        timestamp=first_row["timestamp"],
        value=float(first_row[target_col]) if pd.notna(first_row[target_col]) else 0.0,
        series_id="sensor_42",
    )
    log.info("First observation : %s", obs)
    log.info("  → epoch ms : %s", obs.timestamp_ms)
    log.info("  → UTC dt   : %s", obs.datetime_utc)

    return raw_df


# ---------------------------------------------------------------------------
# Section 3 — Data Cleaning & Sanitisation
# ---------------------------------------------------------------------------

def section_clean(raw_df: pd.DataFrame, value_cols: list, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sanitise the raw DataFrame using linear interpolation.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Output of :func:`section_ingest`.
    value_cols : list of str
        Columns to check for NaN values.
    target_col : str
        Column to extract as the primary univariate series.

    Returns
    -------
    clean_df : pd.DataFrame
        Sanitised DataFrame.
    ts : pd.Series
        DatetimeIndex series for the target column.
    """
    _section("3 — Data Cleaning & Sanitisation")
    from aic_ts_suite.cleaning import DataCleaner, sanitize  # noqa: PLC0415

    # Strategy 1: linear interpolation
    clean_df = sanitize(raw_df, strategy="interpolate")
    log.info("NaN after interpolate:\n%s", clean_df[value_cols].isna().sum().to_string())

    # Strategy 2: forward-fill (informational)
    ffill_df = sanitize(raw_df, strategy="ffill")
    log.info("NaN after ffill:\n%s", ffill_df[value_cols].isna().sum().to_string())

    # Timestamp normalisation to epoch ms via the full DataCleaner pipeline
    _tmp_csv = os.path.join(tempfile.gettempdir(), "demo_sensor.csv")
    raw_df.to_csv(_tmp_csv, index=False)
    pipeline_df = (
        DataCleaner(_tmp_csv, timestamp_col="timestamp")
        .load()
        .sanitize("interpolate")
        .to_epoch_ms()
        .result()
    )
    log.info("Pipeline (epoch ms) head:\n%s", pipeline_df[["timestamp", "timestamp_ms"] + value_cols].head().to_string())

    ts = clean_df.set_index("timestamp")[target_col]
    ts.name = target_col
    return clean_df, ts


# ---------------------------------------------------------------------------
# Section 4 — Variance-Stabilising Transforms
# ---------------------------------------------------------------------------

def section_transforms(ts: pd.Series, no_plots: bool) -> None:
    """
    Apply Log, Sqrt, and Box-Cox transforms and verify round-trip accuracy.

    Parameters
    ----------
    ts : pd.Series
        Cleaned univariate series.
    no_plots : bool
        When True, skip all matplotlib rendering.
    """
    _section("4 — Variance-Stabilising Transforms")
    from aic_ts_suite.signals.transforms import BoxCoxTransform, LogTransform, SqrtTransform  # noqa: PLC0415

    log_t = LogTransform()
    sqrt_t = SqrtTransform()
    bc_t = BoxCoxTransform()

    ts_log = log_t.apply(ts)
    ts_sqrt = sqrt_t.apply(ts)
    ts_bc = bc_t.apply(ts)

    log.info("Box-Cox λ (MLE) = %.4f", bc_t.lmbda)

    # Round-trip accuracy
    ts_recovered = bc_t.inverse(ts_bc)
    roundtrip_error = (ts - ts_recovered).abs().max()
    log.info("Box-Cox round-trip max error: %.2e", roundtrip_error)

    if not no_plots:
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
        for ax, transformed, label in zip(
            axes, [ts_log, ts_sqrt, ts_bc], ["Log", "Sqrt", "Box-Cox"]
        ):
            ax.plot(transformed, linewidth=0.8)
            ax.set_title(f"{label} Transform")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Section 5 — Exploratory Data Analysis
# ---------------------------------------------------------------------------

def section_eda(ts: pd.Series, seasonal_period: int, target_col: str, no_plots: bool) -> None:
    """
    Produce seasonal overlay, ACF/PACF, and classical decomposition plots.

    Parameters
    ----------
    ts : pd.Series
        Cleaned univariate series with a DatetimeIndex.
    seasonal_period : int
        Dominant seasonal period (e.g. 12 for monthly data).
    target_col : str
        Series label used in plot titles.
    no_plots : bool
        When True, skip all matplotlib rendering.
    """
    _section("5 — Exploratory Data Analysis")
    from aic_ts_suite.viz import plot_acf_pacf, plot_decomposition, plot_seasonal  # noqa: PLC0415

    if no_plots:
        log.info("Skipping EDA plots (--no-plots)")
        return

    plot_seasonal(ts, period=seasonal_period, freq_label="Period",
                  title=f"Seasonal Overlay ({seasonal_period}-period cycles)")
    plt.show()

    plot_acf_pacf(ts, lags=3 * seasonal_period, title_prefix=f"{target_col} – ")
    plt.show()

    plot_decomposition(ts, period=seasonal_period, model="additive",
                       title=f"{target_col} – Additive Decomposition")
    plt.show()


# ---------------------------------------------------------------------------
# Section 6 — Feature Engineering
# ---------------------------------------------------------------------------

def section_features(ts: pd.Series, seasonal_period: int, no_plots: bool) -> None:
    """
    Build Fourier harmonic terms, moving averages, and supervised ML matrices.

    Parameters
    ----------
    ts : pd.Series
        Cleaned univariate series.
    seasonal_period : int
        Dominant seasonal period — controls lag depth and Fourier order.
    no_plots : bool
        When True, skip all matplotlib rendering.
    """
    _section("6 — Feature Engineering")
    from aic_ts_suite.features import (  # noqa: PLC0415
        build_supervised_matrix,
        centered_moving_average,
        fourier_terms,
        lag_features,
        optimal_k,
        rolling_lag_features,
        trailing_moving_average,
    )

    # Fourier harmonic terms
    K_star = optimal_k(ts, period=seasonal_period, max_K=seasonal_period // 2)
    log.info("Optimal K for period=%d: %d", seasonal_period, K_star)

    ft = fourier_terms(n=len(ts), period=seasonal_period, K=K_star)
    ft.index = ts.index

    # Moving averages
    trail_3 = trailing_moving_average(ts, window=3)
    trail_sp = trailing_moving_average(ts, window=seasonal_period)
    center_sp = centered_moving_average(ts, window=seasonal_period)

    # Lag / rolling features
    lags_df = lag_features(ts, lags=seasonal_period // 2)
    log.info("Lag features shape : %s", lags_df.shape)

    roll_df = rolling_lag_features(ts, windows=[3, 6, seasonal_period])
    log.info("Rolling features shape : %s", roll_df.shape)

    # Full supervised matrix
    sup = build_supervised_matrix(
        ts,
        lags=seasonal_period,
        rolling_windows=[3, 6, seasonal_period],
        fourier_k=4,
        seasonal_period=seasonal_period,
    )
    log.info("Supervised matrix: %d rows × %d columns", *sup.shape)
    log.info("Columns: %s", list(sup.columns))

    if not no_plots:
        fig, ax = plt.subplots(figsize=(12, 3))
        ft.plot(ax=ax, linewidth=0.7, alpha=0.8)
        ax.set_title(f"Fourier Harmonic Pairs (K={K_star}, period={seasonal_period})")
        ax.legend(ncol=K_star * 2, fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ts, label="Observed", alpha=0.5, linewidth=0.8)
        ax.plot(trail_3, label="Trailing MA(3)", linewidth=1)
        ax.plot(trail_sp, label=f"Trailing MA({seasonal_period})", linewidth=1)
        ax.plot(center_sp, label=f"Centered MA({seasonal_period})", linewidth=1, linestyle="--")
        ax.legend()
        ax.set_title("Moving Averages")
        ax.grid(True, alpha=0.3)
        plt.show()


# ---------------------------------------------------------------------------
# Section 7 — Univariate Forecasting
# ---------------------------------------------------------------------------

def section_univariate(ts: pd.Series, horizon: int, seasonal_period: int, freq: str) -> dict:
    """
    Fit AutoARIMA, AutoETS, Holt-Winters, XGBoost, and Prophet on the train
    split and generate horizon-step-ahead forecasts for the test split.

    Parameters
    ----------
    ts : pd.Series
        Full cleaned univariate series.
    horizon : int
        Number of forecast steps (test set size).
    seasonal_period : int
        Dominant seasonal period passed to each model.
    freq : str
        Pandas DateOffset alias (e.g. 'D', 'MS', 'H').

    Returns
    -------
    dict
        Mapping from model name to ForecastResult, plus 'train' and 'test' keys.
    """
    _section("7 — Univariate Forecasting")
    from aic_ts_suite.forecasting import (  # noqa: PLC0415
        AutoARIMAForecaster,
        AutoETSForecaster,
        HoltWintersForecaster,
        ProphetForecaster,
        XGBoostForecaster,
        auto_select_univariate,
    )

    train, test = ts.iloc[:-horizon], ts.iloc[-horizon:]
    log.info(
        "Train: %s → %s  (%d obs)", train.index[0].date(), train.index[-1].date(), len(train)
    )
    log.info(
        "Test : %s → %s  (%d obs)", test.index[0].date(), test.index[-1].date(), len(test)
    )

    results = {"train": train, "test": test}

    # AutoARIMA
    log.info("Fitting AutoARIMA …")
    arima = AutoARIMAForecaster(seasonal=True, m=seasonal_period, trace=True)
    res_arima = arima.fit_predict(train, horizon, start=test.index[0], freq=freq)
    res_arima.forecast.index = test.index
    if res_arima.lower is not None:
        res_arima.lower.index = test.index
        res_arima.upper.index = test.index
    log.info(
        "AutoARIMA  AICc=%.2f  (%d ms)",
        res_arima.info_criteria["AICc"],
        res_arima.duration_ms,
    )
    results["arima"] = res_arima

    # AutoETS
    log.info("Fitting AutoETS …")
    ets = AutoETSForecaster(seasonal_periods=seasonal_period)
    res_ets = ets.fit_predict(train, horizon)
    res_ets.forecast.index = test.index
    log.info(
        "AutoETS    AICc=%.2f  (%d ms)",
        res_ets.info_criteria.get("AICc", float("nan")),
        res_ets.duration_ms,
    )
    results["ets"] = res_ets

    # Holt-Winters
    log.info("Fitting Holt-Winters …")
    hw = HoltWintersForecaster(seasonal="add", seasonal_periods=seasonal_period)
    res_hw = hw.fit_predict(train, horizon)
    res_hw.forecast.index = test.index
    log.info(
        "Holt-Wint  AICc=%.2f  (%d ms)",
        res_hw.info_criteria.get("AICc", float("nan")),
        res_hw.duration_ms,
    )
    results["hw"] = res_hw

    # Auto-selection via AICc minimisation
    log.info("Running auto_select_univariate …")
    best = auto_select_univariate(
        train, horizon,
        seasonal_periods=seasonal_period,
        start=test.index[0],
        freq=freq,
    )
    best.forecast.index = test.index
    if best.lower is not None:
        best.lower.index = test.index
        best.upper.index = test.index
    log.info(
        "Best model: %s  AICc=%.2f",
        best.model_name,
        best.info_criteria.get("AICc", float("nan")),
    )
    results["best"] = best

    # XGBoost
    log.info("Fitting XGBoostForecaster …")
    xgb_fc = XGBoostForecaster(
        lags=seasonal_period,
        rolling_windows=[3, 6, seasonal_period],
        fourier_k=4,
        fourier_period=seasonal_period,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
    )
    res_xgb = xgb_fc.fit_predict(train, horizon)
    res_xgb.forecast.index = test.index
    if res_xgb.lower is not None:
        res_xgb.lower.index = test.index
        res_xgb.upper.index = test.index
    xgb_rmse = ((test - res_xgb.forecast) ** 2).mean() ** 0.5
    log.info("XGBoost  RMSE=%.3f  (%d ms)", xgb_rmse, res_xgb.duration_ms)
    results["xgb"] = res_xgb

    # Prophet
    log.info("Fitting ProphetForecaster …")
    prophet_fc = ProphetForecaster(
        growth="linear",
        seasonality_mode="additive",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    res_prophet = prophet_fc.fit_predict(train, horizon)
    res_prophet.forecast.index = test.index
    if res_prophet.lower is not None:
        res_prophet.lower.index = test.index
        res_prophet.upper.index = test.index
    prophet_rmse = ((test - res_prophet.forecast) ** 2).mean() ** 0.5
    log.info("Prophet  RMSE=%.3f  (%d ms)", prophet_rmse, res_prophet.duration_ms)
    results["prophet"] = res_prophet

    return results


# ---------------------------------------------------------------------------
# Section 8 — Multivariate Forecasting (VAR)
# ---------------------------------------------------------------------------

def section_var(
    clean_df: pd.DataFrame,
    value_cols: list,
    horizon: int,
    test_index,
) -> object:
    """
    Fit a Vector Autoregression (VAR) model when ≥2 feature columns are available.

    Parameters
    ----------
    clean_df : pd.DataFrame
        Cleaned DataFrame with a 'timestamp' column.
    value_cols : list of str
        Multivariate feature columns.
    horizon : int
        Forecast horizon (must match the test-set length).
    test_index : pd.DatetimeIndex
        Index used to align VAR forecasts with univariate results.

    Returns
    -------
    ForecastResult or None
        VAR forecast result, or None when fewer than 2 features are provided.
    """
    _section("8 — Multivariate Forecasting (VAR)")
    if len(value_cols) < 2:
        log.warning("VAR requires ≥2 features — skipping (only %d provided).", len(value_cols))
        return None

    from aic_ts_suite.forecasting import VARForecaster  # noqa: PLC0415

    multi_df = clean_df.set_index("timestamp")[value_cols]
    multi_train = multi_df.iloc[:-horizon]

    log.info("Fitting VARForecaster (maxlags=12) …")
    var = VARForecaster(maxlags=12, ic="aic")
    res_var = var.fit_predict(multi_train, horizon)
    res_var.forecast.index = test_index

    log.info("VAR lag order: %s", res_var.extra["lag_order"])
    log.info(
        "VAR AIC=%.2f  BIC=%.2f",
        res_var.info_criteria["AIC"],
        res_var.info_criteria["BIC"],
    )
    log.info("Granger Causality (significant at α=0.05):")
    for pair, info in res_var.extra["granger"].items():
        if info["min_p_value"] < 0.05:
            log.info("  %s (p=%.4f)", pair, info["min_p_value"])

    return res_var


# ---------------------------------------------------------------------------
# Section 9 & 10 — Evaluation & Leaderboard
# ---------------------------------------------------------------------------

def section_evaluation(
    ts: pd.Series,
    results: dict,
    res_var,
    no_plots: bool,
) -> None:
    """
    Compute KPIs for all models, build a comparison leaderboard, and plot forecasts.

    Parameters
    ----------
    ts : pd.Series
        Full univariate series (train + test).
    results : dict
        Output of :func:`section_univariate` containing model results.
    res_var : ForecastResult or None
        VAR result from :func:`section_var`.
    no_plots : bool
        When True, skip all matplotlib rendering.
    """
    _section("9 & 10 — KPI Evaluation & Leaderboard")
    from aic_ts_suite.evaluation import ModelComparison, compute_all_kpis  # noqa: PLC0415
    from aic_ts_suite.viz import plot_forecast  # noqa: PLC0415

    test = results["test"]
    train = results["train"]

    # Forecast plot for the auto-selected best model
    if not no_plots:
        kpis_arima = compute_all_kpis(test, results["arima"].forecast)
        plot_forecast(
            observed=ts,
            forecast=results["arima"].forecast,
            lower=results["arima"].lower,
            upper=results["arima"].upper,
            train_end=train.index[-1],
            kpis=kpis_arima,
            title="AutoARIMA – Observed vs Forecast",
        )
        plt.show()

    # Model comparison leaderboard
    cmp = ModelComparison(observed_test=test)
    cmp.add(results["arima"])
    cmp.add(results["ets"])
    cmp.add(results["hw"])
    cmp.add(results["xgb"])
    cmp.add(results["prophet"])
    if res_var is not None:
        cmp.add(res_var)

    leaderboard = cmp.leaderboard(sort_by="RMSE")
    log.info("Leaderboard:\n%s", leaderboard.to_string())

    deltas = cmp.metric_deltas(baseline_model=leaderboard.iloc[0]["Model"])
    log.info("Metric deltas vs best:\n%s", deltas.to_string())

    if not no_plots:
        cmp.plot_all(observed_full=ts, train_end=train.index[-1])
        plt.show()


# ---------------------------------------------------------------------------
# Section 11 — Styled Display & Metric Cards
# ---------------------------------------------------------------------------

def section_display(ts: pd.Series, results: dict, config) -> None:
    """
    Render a styled HTML summary table and metric cards for the best model.

    Parameters
    ----------
    ts : pd.Series
        Full univariate series (used for descriptive statistics).
    results : dict
        Must contain 'best' and 'test' keys from :func:`section_univariate`.
    config : CONFIG
        Global config object (provides correlation_id).
    """
    _section("11 — Styled Display & Metric Cards")
    from aic_ts_suite.display import metric_cards, styled_summary  # noqa: PLC0415
    from aic_ts_suite.evaluation import compute_all_kpis  # noqa: PLC0415

    styled_summary(ts.describe().to_frame(), caption=f"{ts.name} – Descriptive Statistics")

    best = results["best"]
    test = results["test"]
    best_kpis = compute_all_kpis(test, best.forecast)
    best_kpis["Model"] = best.model_name
    best_kpis["Duration (ms)"] = best.duration_ms
    best_kpis["correlationId"] = config.correlation_id[:8] + "…"
    metric_cards(best_kpis)


# ---------------------------------------------------------------------------
# Section 12 — Run Traceability
# ---------------------------------------------------------------------------

def section_traceability(results: dict, res_var, config) -> None:
    """
    Log every model run via AnalyticsEngineClient and print the JSON summary.

    Parameters
    ----------
    results : dict
        Output of :func:`section_univariate`.
    res_var : ForecastResult or None
        VAR result (or None to skip).
    config : CONFIG
        Global config object (provides correlation_id).
    """
    _section("12 — Run Traceability")
    from aic_ts_suite.evaluation import AnalyticsEngineClient  # noqa: PLC0415

    client = AnalyticsEngineClient()
    test = results["test"]
    for key in ("arima", "ets", "hw", "xgb", "prophet"):
        client.log_run(results[key], test)
    if res_var is not None:
        client.log_run(res_var, test)

    client.summary()
    log.info("JSON export:\n%s", client.to_json())
    log.info("Session correlationId: %s", config.correlation_id)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    """
    Orchestrate the full toolkit demonstration pipeline.

    Parses CLI arguments, configures logging, and executes each section in
    the order documented in the module docstring.
    """
    args = parse_args(argv)

    # --- Logging setup ---
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=level,
    )

    # --- Matplotlib backend ---
    if args.no_plots:
        matplotlib.use("Agg")

    log.info("Starting toolkit_demo")
    log.info("  csv-path    : %s", args.csv_path)
    log.info("  value-cols  : %s", args.value_cols)
    log.info("  target-col  : %s", args.target_col)
    log.info("  horizon     : %d", args.horizon)
    log.info("  seasonal    : %d", args.seasonal_period)
    log.info("  freq        : %s", args.freq)

    # Section 1
    config = section_config(args.verbose)

    # Section 2
    raw_df = section_ingest(
        args.csv_path, args.timestamp_col, args.value_cols, args.target_col
    )

    # Section 3
    clean_df, ts = section_clean(raw_df, args.value_cols, args.target_col)

    # Section 4
    section_transforms(ts, args.no_plots)

    # Section 5
    section_eda(ts, args.seasonal_period, args.target_col, args.no_plots)

    # Section 6
    section_features(ts, args.seasonal_period, args.no_plots)

    # Sections 7a–c
    results = section_univariate(ts, args.horizon, args.seasonal_period, args.freq)

    # Section 8
    res_var = section_var(clean_df, args.value_cols, args.horizon, results["test"].index)

    # Sections 9–10
    section_evaluation(ts, results, res_var, args.no_plots)

    # Section 11
    section_display(ts, results, config)

    # Section 12
    section_traceability(results, res_var, config)

    log.info("Done.")


if __name__ == "__main__":
    main()
