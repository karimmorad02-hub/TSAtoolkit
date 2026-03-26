#!/usr/bin/env python3
"""
toolkit_demo.py — aic_ts_suite experiment runner.

Driven entirely by a parameters YAML file. Each invocation creates an isolated
experiment folder containing checkpoints, per-model run artefacts, structured
logs, and a final leaderboard.

Folder layout (created automatically)
──────────────────────────────────────
experiments/
  {name}_{YYYYMMDD}_{HHMMSS}_{run_id}/
    params.yaml                         ← verbatim copy of input config
    checkpoints/
      best_checkpoints.json             ← top-N registry (RMSE-ranked)
      checkpoint_{rank}_{model}_{score:.4f}_{ts}.pkl
    runs/
      {model}_run.json                  ← forecast values + metrics
    test_runs/
      {model}_test.json                 ← held-out test evaluation
    logs/
      {YYYYMMDD}_{HHMMSS}_{name}_{model}_{run_id}.log
    results/
      leaderboard.json
      summary.json

Usage
─────
  # Fresh run:
  python scripts/toolkit_demo.py --params configs/params.yaml

  # Resume an interrupted run (reloads completed model checkpoints):
  python scripts/toolkit_demo.py --params configs/params.yaml \\
      --resume experiments/energy_forecast_20260324_143200_a3f2

  # Verbose console output:
  python scripts/toolkit_demo.py --params configs/params.yaml --verbose
"""

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ORDER = ["arima", "ets", "holt_winters", "xgboost", "prophet", "var"]
CHECKPOINT_REGISTRY = "best_checkpoints.json"


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        prog="toolkit_demo",
        description="aic_ts_suite experiment runner — driven by a params YAML file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--params", "-p",
        required=True,
        metavar="YAML",
        help="Path to experiment parameters YAML file (required).",
    )
    p.add_argument(
        "--resume",
        metavar="EXPERIMENT_DIR",
        default=None,
        help="Path to an existing experiment folder to resume from.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Stream INFO-level logs to the console.",
    )
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# YAML loading & validation
# ─────────────────────────────────────────────────────────────────────────────

def load_params(path: str) -> dict:
    """Load and validate the parameters YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)

    # Required top-level keys
    for key in ("experiment", "data", "modelling", "models"):
        if key not in cfg:
            raise ValueError(f"params.yaml is missing required section: '{key}'")

    # Evaluate any arithmetic expressions in numeric modelling fields
    for field in ("horizon", "seasonal_period"):
        val = cfg["modelling"].get(field)
        if isinstance(val, str):
            try:
                cfg["modelling"][field] = int(eval(val, {"__builtins__": {}}))
            except Exception:
                raise ValueError(
                    f"modelling.{field} could not be evaluated as a number: {val!r}"
                )

    # Data validation
    csv = Path(cfg["data"]["csv_path"])
    if not csv.exists():
        raise FileNotFoundError(f"CSV not found: {csv}")
    if cfg["modelling"]["horizon"] <= 0:
        raise ValueError("modelling.horizon must be > 0")
    if not any(v.get("enabled") for v in cfg["models"].values()):
        raise ValueError("At least one model must be enabled in params.yaml")

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Experiment directory setup
# ─────────────────────────────────────────────────────────────────────────────

def make_experiment_dir(cfg: dict, resume_dir: Optional[str]) -> Path:
    """
    Create (or reuse) the experiment output directory.

    Returns the Path to the experiment root.
    """
    if resume_dir:
        exp_dir = Path(resume_dir)
        if not exp_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {exp_dir}")
        return exp_dir

    name = cfg["experiment"]["name"].replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    exp_dir = Path("experiments") / f"{name}_{ts}_{run_id}"

    for subdir in ("checkpoints", "runs", "test_runs", "logs", "results", "plots"):
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Persist a verbatim copy of the config
    shutil.copy(cfg["_source_path"], exp_dir / "params.yaml")
    return exp_dir


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def make_model_logger(
    exp_dir: Path,
    model_name: str,
    cfg: dict,
    run_id: str,
) -> logging.Logger:
    """
    Create a dedicated logger for one model.

    Log filename: {YYYYMMDD}_{HHMMSS}_{experiment}_{model}_{run_id}.log
    Log line format: {ISO_TS} | {LEVEL} | {experiment} | {model} | {run_id} | {msg}

    The filename encodes all searchable parameters so you can grep/ls to find
    the exact run you want without opening any file.
    """
    exp_name = cfg["experiment"]["name"]
    level_name = cfg.get("logging", {}).get("level", "INFO")
    level = getattr(logging, level_name, logging.INFO)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{ts}_{exp_name}_{model_name}_{run_id}.log"
    log_path = exp_dir / "logs" / log_filename

    logger = logging.getLogger(f"{exp_name}.{model_name}.{run_id}")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        f"%(asctime)s | %(levelname)s | {exp_name} | {model_name} | {run_id} | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if cfg.get("logging", {}).get("file", True):
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger, log_path


def make_console_logger(verbose: bool, cfg: dict) -> logging.Logger:
    """Global console logger; only active when --verbose is set."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)
    return logging.getLogger("toolkit_demo")


def section(log: logging.Logger, title: str) -> None:
    log.info("")
    log.info("=" * 70)
    log.info("  %s", title)
    log.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint management
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint_registry(exp_dir: Path) -> list:
    """Load the best-checkpoints registry, or return an empty list."""
    reg_path = exp_dir / "checkpoints" / CHECKPOINT_REGISTRY
    if reg_path.exists():
        with reg_path.open() as f:
            return json.load(f)
    return []


def save_checkpoint_registry(exp_dir: Path, registry: list) -> None:
    reg_path = exp_dir / "checkpoints" / CHECKPOINT_REGISTRY
    with reg_path.open("w") as f:
        json.dump(registry, f, indent=2)


def save_checkpoint(
    exp_dir: Path,
    model_name: str,
    result,
    kpis: dict,
    keep_best_n: int,
    metric: str = "RMSE",
) -> list:
    """
    Serialise a ForecastResult, update the top-N registry, and prune stale files.

    Returns the updated registry list.
    """
    registry = load_checkpoint_registry(exp_dir)
    score = kpis.get(metric, float("inf"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"checkpoint_{model_name}_{score:.4f}_{ts}.pkl"
    ckpt_path = exp_dir / "checkpoints" / filename

    with ckpt_path.open("wb") as f:
        pickle.dump({"model": model_name, "result": result, "kpis": kpis}, f)

    entry = {
        "model": model_name,
        "score": score,
        "metric": metric,
        "file": filename,
        "timestamp": ts,
    }
    registry.append(entry)

    # Sort ascending by score (lower RMSE = better) and keep top-N
    registry.sort(key=lambda x: x["score"])
    evicted = registry[keep_best_n:]
    registry = registry[:keep_best_n]

    for e in evicted:
        stale = exp_dir / "checkpoints" / e["file"]
        if stale.exists():
            stale.unlink()

    save_checkpoint_registry(exp_dir, registry)
    return registry


def sprint_checkpoint(exp_dir: Path, completed_models: list, sprint_num: int) -> None:
    """Save a named sprint snapshot listing completed models at this point."""
    sprint_path = exp_dir / "checkpoints" / f"sprint_{sprint_num:03d}.json"
    with sprint_path.open("w") as f:
        json.dump({
            "sprint": sprint_num,
            "timestamp": datetime.now().isoformat(),
            "completed_models": completed_models,
        }, f, indent=2)


def load_completed_models(exp_dir: Path) -> dict:
    """
    Scan runs/ for completed model JSON files.

    Returns {model_name: kpis_dict} for all already-finished models.
    """
    completed = {}
    runs_dir = exp_dir / "runs"
    if not runs_dir.exists():
        return completed
    for f in runs_dir.glob("*_run.json"):
        model = f.stem.replace("_run", "")
        with f.open() as fp:
            completed[model] = json.load(fp)
    return completed


# ─────────────────────────────────────────────────────────────────────────────
# Run artefact helpers
# ─────────────────────────────────────────────────────────────────────────────

def _series_to_list(s) -> list:
    """Safely convert a pandas Series (or None) to a JSON-serialisable list."""
    if s is None:
        return None
    return [None if (v is not None and np.isnan(v)) else v for v in s.tolist()]


def save_run(exp_dir: Path, model_name: str, result, kpis: dict) -> None:
    """Write forecast values + KPIs to runs/{model}_run.json."""
    payload = {
        "model": model_name,
        "kpis": kpis,
        "duration_ms": getattr(result, "duration_ms", None),
        "forecast": _series_to_list(getattr(result, "forecast", None)),
        "lower": _series_to_list(getattr(result, "lower", None)),
        "upper": _series_to_list(getattr(result, "upper", None)),
        "info_criteria": getattr(result, "info_criteria", {}),
    }
    out = exp_dir / "runs" / f"{model_name}_run.json"
    with out.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def save_test_run(exp_dir: Path, model_name: str, test: pd.Series, forecast: pd.Series) -> None:
    """Write per-step test actuals vs forecast to test_runs/{model}_test.json."""
    records = [
        {"timestamp": str(ts), "actual": float(a), "forecast": float(fc)}
        for ts, a, fc in zip(test.index, test.values, forecast.values)
    ]
    out = exp_dir / "test_runs" / f"{model_name}_test.json"
    with out.open("w") as f:
        json.dump(records, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Data pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_data_pipeline(cfg: dict, log: logging.Logger) -> tuple[pd.DataFrame, pd.Series]:
    """Load, subsample, clean, optionally enrich with weather, and return (clean_df, target_series)."""
    from aic_ts_suite.cleaning import sanitize

    dc = cfg["data"]
    section(log, "Data Ingestion & Cleaning")

    raw = pd.read_csv(dc["csv_path"], parse_dates=[dc["timestamp_col"]])
    raw = (
        raw[[dc["timestamp_col"]] + dc["value_cols"]]
        .rename(columns={dc["timestamp_col"]: "timestamp"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    max_rows = dc.get("max_rows")
    if max_rows:
        raw = raw.tail(int(max_rows)).reset_index(drop=True)
        log.info("Subsampled to last %d rows.", max_rows)

    log.info("Raw shape: %s | NaN counts: %s",
             raw.shape,
             raw[dc["value_cols"]].isna().sum().to_dict())

    clean = sanitize(raw, strategy="interpolate")

    # ── Weather enrichment ────────────────────────────────────────────────────
    wc = cfg.get("weather", {})
    if wc.get("enabled", False):
        clean = _enrich_with_weather(clean, cfg, log)

    ts = clean.set_index("timestamp")[dc["target_col"]]
    ts.name = dc["target_col"]
    log.info("Clean shape: %s", clean.shape)
    return clean, ts


def _enrich_with_weather(
    df: pd.DataFrame,
    cfg: dict,
    log: logging.Logger,
) -> pd.DataFrame:
    """
    Fetch weather data for the date range in *df* and merge it as extra columns.

    Reads the ``weather`` section of *cfg*.  On any error the original
    DataFrame is returned unchanged so the experiment can continue.
    """
    from aic_ts_suite.connectivity.weather import fetch_weather, merge_weather

    wc = cfg["weather"]
    prefix = wc.get("prefix", "weather_")

    # Infer actual data frequency from timestamps so weather resampling
    # matches the raw data granularity (e.g. 15min), not the modelling freq.
    explicit_freq = wc.get("resample_freq")
    if explicit_freq:
        freq = explicit_freq
    else:
        inferred = pd.infer_freq(pd.to_datetime(df["timestamp"]).head(20))
        freq = inferred if inferred else cfg["modelling"].get("freq", "MS")
        log.info("Weather resample freq auto-detected: %s", freq)

    ts_col = df["timestamp"]
    start_date = pd.to_datetime(ts_col.min()).date()
    end_date = pd.to_datetime(ts_col.max()).date()

    log.info(
        "Fetching weather  lat=%.4f lon=%.4f  %s → %s  vars=%s",
        wc["latitude"], wc["longitude"],
        start_date, end_date, wc["variables"],
    )

    try:
        weather_df = fetch_weather(
            latitude=wc["latitude"],
            longitude=wc["longitude"],
            variables=wc["variables"],
            start_date=str(start_date),
            end_date=str(end_date),
            source=wc.get("source", "archive"),
            timeout=wc.get("timeout", 60),
        )
        if weather_df.empty:
            log.warning("Weather fetch returned empty DataFrame — skipping enrichment.")
            return df

        enriched = merge_weather(df, weather_df, freq=freq, prefix=prefix)
        new_cols = [c for c in enriched.columns if c.startswith(prefix)]
        log.info("Weather enrichment complete: added %d columns: %s", len(new_cols), new_cols)
        return enriched

    except Exception as exc:
        log.warning("Weather enrichment failed (%s: %s) — continuing without weather data.",
                    type(exc).__name__, exc)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers for training steps
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, exp_dir: Path, name: str, log: logging.Logger) -> None:
    """Save a matplotlib figure to plots/{name}.png and close it."""
    import matplotlib.pyplot as plt
    out = exp_dir / "plots" / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved → %s", out.name)


def _log_series_stats(log: logging.Logger, series: pd.Series, label: str = "Series") -> None:
    """Log descriptive statistics and date range of a training series."""
    log.info("── %s stats ──────────────────────────────", label)
    log.info("  Observations : %d", len(series))
    log.info("  Date range   : %s → %s", series.index[0], series.index[-1])
    log.info("  Mean         : %.4f", series.mean())
    log.info("  Std          : %.4f", series.std())
    log.info("  Min / Max    : %.4f / %.4f", series.min(), series.max())
    log.info("  NaN count    : %d", series.isna().sum())


def _log_insample(log: logging.Logger, fitted: pd.Series, actual: pd.Series) -> None:
    """Log in-sample fit quality (RMSE, MAE, residual mean/std)."""
    if fitted is None or len(fitted) == 0:
        return
    aligned = actual.reindex(fitted.index).dropna()
    f_aligned = fitted.reindex(aligned.index).dropna()
    if len(aligned) == 0:
        return
    resid = aligned - f_aligned
    rmse = float((resid ** 2).mean() ** 0.5)
    mae = float(resid.abs().mean())
    log.info("── In-sample fit ─────────────────────────")
    log.info("  Fitted obs   : %d", len(f_aligned))
    log.info("  RMSE (train) : %.4f", rmse)
    log.info("  MAE  (train) : %.4f", mae)
    log.info("  Residual μ   : %.4f  σ=%.4f", resid.mean(), resid.std())


def _log_forecast_summary(log: logging.Logger, res, test_index) -> None:
    """Log forecast point statistics and prediction interval width."""
    fc = res.forecast
    log.info("── Forecast summary ──────────────────────")
    log.info("  Steps        : %d", len(fc))
    log.info("  Forecast μ   : %.4f  σ=%.4f", fc.mean(), fc.std())
    log.info("  Forecast range: %.4f → %.4f", fc.min(), fc.max())
    if res.lower is not None and res.upper is not None:
        pi_width = (res.upper - res.lower).mean()
        log.info("  Mean PI width: %.4f  (%.0f%% CI)", pi_width,
                 100 * (1 - (1 - 0.95)))


def _log_step(log: logging.Logger, step: int, total: int, desc: str, *args) -> None:
    msg = desc % args if args else desc
    log.info("STEP %d/%d  %s", step, total, msg)


# ─────────────────────────────────────────────────────────────────────────────
# Model runners
# ─────────────────────────────────────────────────────────────────────────────

def run_arima(train, horizon, cfg, log):
    import time
    from aic_ts_suite.forecasting import AutoARIMAForecaster
    mc = cfg["models"]["arima"]
    m = cfg["modelling"]["seasonal_period"]
    freq = cfg["modelling"]["freq"]

    log.info("━━━ AutoARIMA — hyperparameters ━━━━━━━━━━")
    log.info("  seasonal       : %s", mc["seasonal"])
    log.info("  seasonal period: %d", m)
    log.info("  stepwise search: True")
    log.info("  freq           : %s", freq)
    log.info("  horizon        : %d", horizon)
    _log_series_stats(log, train, "Train")

    _log_step(log, 1, 3, "Initialising AutoARIMA candidate search …")
    f = AutoARIMAForecaster(seasonal=mc["seasonal"], m=m, trace=False)

    _log_step(log, 2, 3, "Fitting best ARIMA/SARIMA order via stepwise AICc …")
    t0 = time.perf_counter()
    f.fit(train, start=train.index[-1], freq=freq)
    fit_ms = (time.perf_counter() - t0) * 1000
    order = f._model.order
    seas_order = f._model.seasonal_order
    log.info("  Selected order   : ARIMA%s", order)
    log.info("  Seasonal order   : %s", seas_order)
    log.info("  AICc / AIC / BIC : %.2f / %.2f / %.2f",
             f._model.aicc(), f._model.aic(), f._model.bic())
    log.info("  Fit duration     : %.0f ms", fit_ms)

    _log_step(log, 3, 3, "Generating %d-step-ahead forecast with %.0f%% PI …", horizon, 95)
    t1 = time.perf_counter()
    res = f.predict(horizon, start=train.index[-1], freq=freq)
    predict_ms = (time.perf_counter() - t1) * 1000
    res.duration_ms = fit_ms + predict_ms

    _log_insample(log, res.fitted_values, train)
    _log_forecast_summary(log, res, None)
    log.info("── Timing ────────────────────────────────")
    log.info("  Fit     : %.0f ms", fit_ms)
    log.info("  Predict : %.0f ms", predict_ms)
    log.info("  Total   : %.0f ms", res.duration_ms)
    return res


def run_ets(train, horizon, cfg, log):
    import time
    from aic_ts_suite.forecasting import AutoETSForecaster
    m = cfg["modelling"]["seasonal_period"]
    n_candidates = 2 * 3 * 3  # errors × trends × seasons

    log.info("━━━ AutoETS — hyperparameters ━━━━━━━━━━━━")
    log.info("  seasonal_periods: %d", m)
    log.info("  search space    : %d ETS configurations", n_candidates)
    log.info("  selection metric: AIC (lower = better)")
    log.info("  horizon         : %d", horizon)
    _log_series_stats(log, train, "Train")

    _log_step(log, 1, 2,
              "Grid search over %d ETS(error, trend, seasonal) configurations …", n_candidates)
    f = AutoETSForecaster(seasonal_periods=m)
    t0 = time.perf_counter()
    f.fit(train)
    fit_ms = (time.perf_counter() - t0) * 1000

    # Extract selected config from the fitted model
    fitted_model = f._fit
    ets_model = f._model
    log.info("  Selected config  : error=%s  trend=%s  seasonal=%s",
             getattr(ets_model, 'error', '?'),
             getattr(ets_model, 'trend', '?'),
             getattr(ets_model, 'seasonal', '?'))
    log.info("  AICc / AIC / BIC : %.2f / %.2f / %.2f",
             fitted_model.aicc, fitted_model.aic, fitted_model.bic)
    log.info("  Fit duration     : %.0f ms", fit_ms)

    _log_step(log, 2, 2, "Generating %d-step-ahead forecast …", horizon)
    t1 = time.perf_counter()
    res = f.predict(horizon)
    predict_ms = (time.perf_counter() - t1) * 1000
    res.duration_ms = fit_ms + predict_ms

    _log_insample(log, res.fitted_values, train)
    _log_forecast_summary(log, res, None)
    log.info("── Timing ────────────────────────────────")
    log.info("  Fit     : %.0f ms", fit_ms)
    log.info("  Predict : %.0f ms", predict_ms)
    log.info("  Total   : %.0f ms", res.duration_ms)
    return res


def run_holt_winters(train, horizon, cfg, log):
    import time
    from aic_ts_suite.forecasting import HoltWintersForecaster
    mc = cfg["models"]["holt_winters"]
    m = cfg["modelling"]["seasonal_period"]

    log.info("━━━ Holt-Winters — hyperparameters ━━━━━━")
    log.info("  trend          : additive")
    log.info("  seasonal       : %s", mc["seasonal"])
    log.info("  seasonal_periods: %d", m)
    log.info("  optimised      : True (L-BFGS-B)")
    log.info("  PI method      : simulation (500 reps)")
    log.info("  horizon        : %d", horizon)
    _log_series_stats(log, train, "Train")

    _log_step(log, 1, 2, "Fitting Holt-Winters model (optimising α, β, γ) …")
    f = HoltWintersForecaster(seasonal=mc["seasonal"], seasonal_periods=m)
    t0 = time.perf_counter()
    f.fit(train)
    fit_ms = (time.perf_counter() - t0) * 1000

    hw = f._fit
    params = hw.params
    log.info("  α (level)      : %.4f", params.get("smoothing_level", float("nan")))
    log.info("  β (trend)      : %.4f", params.get("smoothing_trend", float("nan")))
    log.info("  γ (seasonal)   : %.4f", params.get("smoothing_seasonal", float("nan")))
    log.info("  AICc / AIC / BIC : %.2f / %.2f / %.2f", hw.aicc, hw.aic, hw.bic)
    log.info("  Fit duration   : %.0f ms", fit_ms)

    _log_step(log, 2, 2,
              "Generating %d-step forecast + PI via 500-rep simulation …", horizon)
    t1 = time.perf_counter()
    res = f.predict(horizon)
    predict_ms = (time.perf_counter() - t1) * 1000
    res.duration_ms = fit_ms + predict_ms

    _log_insample(log, res.fitted_values, train)
    _log_forecast_summary(log, res, None)
    log.info("── Timing ────────────────────────────────")
    log.info("  Fit     : %.0f ms", fit_ms)
    log.info("  Predict : %.0f ms", predict_ms)
    log.info("  Total   : %.0f ms", res.duration_ms)
    return res


def run_xgboost(train, horizon, cfg, log):
    import time
    from aic_ts_suite.forecasting import XGBoostForecaster
    mc = cfg["models"]["xgboost"]
    m = cfg["modelling"]["seasonal_period"]
    lags = mc.get("lags", m)
    rolling = mc.get("rolling_windows", [3, 6, m])
    fourier_k = mc.get("fourier_k", 4)
    n_features_est = lags + len(rolling) * 2 + fourier_k * 2

    log.info("━━━ XGBoost — hyperparameters ━━━━━━━━━━━")
    log.info("  n_estimators   : %d", mc["n_estimators"])
    log.info("  learning_rate  : %.4f", mc["learning_rate"])
    log.info("  max_depth      : %d", mc["max_depth"])
    log.info("  lags           : %d  (lag_1 … lag_%d)", lags, lags)
    log.info("  rolling_windows: %s  (mean + std each)", rolling)
    log.info("  fourier_k      : %d pairs  (period=%d)", fourier_k, m)
    log.info("  est. features  : ~%d", n_features_est)
    log.info("  PI method      : quantile regression (3 models)")
    log.info("  horizon        : %d (recursive)", horizon)
    _log_series_stats(log, train, "Train")

    f = XGBoostForecaster(
        lags=lags,
        rolling_windows=rolling,
        fourier_k=fourier_k,
        fourier_period=m,
        n_estimators=mc["n_estimators"],
        learning_rate=mc["learning_rate"],
        max_depth=mc["max_depth"],
    )

    _log_step(log, 1, 3,
              "Building supervised lag/rolling/Fourier feature matrix …")
    t0 = time.perf_counter()
    f.fit(train)
    fit_ms = (time.perf_counter() - t0) * 1000
    actual_features = len(f._feature_cols) if f._feature_cols else n_features_est
    log.info("  Feature matrix : %d rows × %d features (after drop_na)",
             len(train), actual_features)
    log.info("  Feature names  : %s … (top 5)", f._feature_cols[:5] if f._feature_cols else [])

    _log_step(log, 2, 3,
              "Fit complete — logging top-10 feature importances …")
    try:
        imp = f.feature_importance().head(10)
        for feat, score in imp.items():
            log.info("    %-25s  %.4f", feat, score)
    except Exception:
        log.info("  (feature importances unavailable)")
    log.info("  Fit duration   : %.0f ms", fit_ms)

    _log_step(log, 3, 3,
              "Recursive %d-step forecast (point + quantile PI) …", horizon)
    t1 = time.perf_counter()
    res = f.predict(horizon)
    predict_ms = (time.perf_counter() - t1) * 1000
    res.duration_ms = fit_ms + predict_ms

    _log_insample(log, res.fitted_values, train)
    _log_forecast_summary(log, res, None)
    log.info("── Timing ────────────────────────────────")
    log.info("  Feature build + fit : %.0f ms", fit_ms)
    log.info("  Recursive predict   : %.0f ms", predict_ms)
    log.info("  Total               : %.0f ms", res.duration_ms)
    return res


def run_prophet(train, horizon, cfg, log):
    import time
    from aic_ts_suite.forecasting import ProphetForecaster
    mc = cfg["models"]["prophet"]

    log.info("━━━ Prophet — hyperparameters ━━━━━━━━━━━")
    log.info("  growth                 : %s", mc.get("growth", "linear"))
    log.info("  seasonality_mode       : %s", mc.get("seasonality_mode", "additive"))
    log.info("  yearly_seasonality     : %s", mc.get("yearly_seasonality", True))
    log.info("  weekly_seasonality     : %s", mc.get("weekly_seasonality", False))
    log.info("  daily_seasonality      : %s", mc.get("daily_seasonality", False))
    log.info("  changepoint_prior_scale: %.4f", mc.get("changepoint_prior_scale", 0.05))
    log.info("  horizon                : %d", horizon)
    _log_series_stats(log, train, "Train")

    f = ProphetForecaster(
        growth=mc.get("growth", "linear"),
        seasonality_mode=mc.get("seasonality_mode", "additive"),
        yearly_seasonality=mc.get("yearly_seasonality", True),
        weekly_seasonality=mc.get("weekly_seasonality", False),
        daily_seasonality=mc.get("daily_seasonality", False),
        changepoint_prior_scale=mc.get("changepoint_prior_scale", 0.05),
    )

    _log_step(log, 1, 2,
              "Fitting Prophet — trend + seasonality decomposition (MAP) …")
    t0 = time.perf_counter()
    f.fit(train)
    fit_ms = (time.perf_counter() - t0) * 1000
    n_changepoints = len(f._model.changepoints) if f._model else 0
    log.info("  Changepoints detected: %d", n_changepoints)
    if n_changepoints > 0 and f._model:
        log.info("  First changepoint    : %s", f._model.changepoints.iloc[0])
        log.info("  Last changepoint     : %s", f._model.changepoints.iloc[-1])
    log.info("  Fit duration         : %.0f ms", fit_ms)

    _log_step(log, 2, 2,
              "Generating %d-step forecast (trend + seasonal components) …", horizon)
    t1 = time.perf_counter()
    res = f.predict(horizon)
    predict_ms = (time.perf_counter() - t1) * 1000
    res.duration_ms = fit_ms + predict_ms

    _log_insample(log, res.fitted_values, train)
    _log_forecast_summary(log, res, None)
    log.info("── Timing ────────────────────────────────")
    log.info("  Fit     : %.0f ms", fit_ms)
    log.info("  Predict : %.0f ms", predict_ms)
    log.info("  Total   : %.0f ms", res.duration_ms)
    return res


def run_var(train_full_df, horizon, cfg, log):
    import time
    from aic_ts_suite.forecasting import VARForecaster
    mc = cfg["models"]["var"]
    dc = cfg["data"]
    if len(dc["value_cols"]) < 2:
        log.warning("VAR skipped — needs ≥2 value_cols.")
        return None
    n_vars = len(dc["value_cols"])

    log.info("━━━ VAR — hyperparameters ━━━━━━━━━━━━━━━")
    log.info("  variables      : %d  %s", n_vars, dc["value_cols"])
    log.info("  maxlags        : %d", mc["maxlags"])
    log.info("  ic             : %s", mc["ic"])
    log.info("  Granger maxlag : 12")
    log.info("  horizon        : %d", horizon)
    log.info("── Multivariate train stats ──────────────")
    for col in dc["value_cols"]:
        s = train_full_df[col]
        log.info("  %-45s  μ=%.2f  σ=%.2f  [%.2f, %.2f]",
                 col, s.mean(), s.std(), s.min(), s.max())
    log.info("  Train shape    : %s", train_full_df.shape)
    log.info("  Date range     : %s → %s",
             train_full_df.index[0], train_full_df.index[-1])

    _log_step(log, 1, 3,
              "Fitting VAR(maxlags=%d) — lag order selection via %s …",
              mc["maxlags"], mc["ic"].upper())
    f = VARForecaster(maxlags=mc["maxlags"], ic=mc["ic"])
    t0 = time.perf_counter()
    f.fit(train_full_df)
    fit_ms = (time.perf_counter() - t0) * 1000
    k_ar = f._fit.k_ar
    log.info("  Selected lag order : %d", k_ar)
    log.info("  AIC / BIC          : %.2f / %.2f",
             f._fit.aic, f._fit.bic)
    log.info("  Fit duration       : %.0f ms", fit_ms)

    _log_step(log, 2, 3, "Granger causality results (α=0.05) …")
    sig_pairs = {k: v for k, v in f.granger_results.items() if v["min_p_value"] < 0.05}
    insig_pairs = {k: v for k, v in f.granger_results.items() if v["min_p_value"] >= 0.05}
    log.info("  Significant pairs  : %d / %d total",
             len(sig_pairs), len(f.granger_results))
    for pair, info in sig_pairs.items():
        log.info("    ✓ %s  (p=%.4f)", pair, info["min_p_value"])
    for pair, info in insig_pairs.items():
        log.info("    ✗ %s  (p=%.4f)", pair, info["min_p_value"])

    _log_step(log, 3, 3,
              "Generating %d-step multivariate forecast + bootstrap PI …", horizon)
    t1 = time.perf_counter()
    res = f.predict(horizon)
    predict_ms = (time.perf_counter() - t1) * 1000
    res.duration_ms = fit_ms + predict_ms

    _log_forecast_summary(log, res, None)
    log.info("── Timing ────────────────────────────────")
    log.info("  Fit + Granger : %.0f ms", fit_ms)
    log.info("  Predict       : %.0f ms", predict_ms)
    log.info("  Total         : %.0f ms", res.duration_ms)
    return res


MODEL_FN = {
    "arima": run_arima,
    "ets": run_ets,
    "holt_winters": run_holt_winters,
    "xgboost": run_xgboost,
    "prophet": run_prophet,
    "var": run_var,
}


# ─────────────────────────────────────────────────────────────────────────────
# KPI computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_kpis(actual: pd.Series, forecast: pd.Series) -> dict:
    from aic_ts_suite.evaluation import compute_all_kpis
    return compute_all_kpis(actual, forecast)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(cfg: dict, exp_dir: Path, console_log: logging.Logger) -> None:
    run_id = exp_dir.name.split("_")[-1]
    ckpt_cfg = cfg.get("checkpoints", {})
    keep_best_n = ckpt_cfg.get("keep_best_n", 3)
    metric = ckpt_cfg.get("metric", "RMSE")
    sprint_interval = ckpt_cfg.get("sprint_interval", 2)
    horizon = cfg["modelling"]["horizon"]

    # ── Data ─────────────────────────────────────────────────────────────────
    clean_df, ts = run_data_pipeline(cfg, console_log)
    train, test = ts.iloc[:-horizon], ts.iloc[-horizon:]
    console_log.info("Train=%d obs | Test=%d obs", len(train), len(test))

    # Multivariate train slice
    value_cols = cfg["data"]["value_cols"]
    multi_train = clean_df.set_index("timestamp")[value_cols].iloc[:-horizon]

    # ── Resume: reload already-completed models ───────────────────────────────
    completed = load_completed_models(exp_dir)
    if completed:
        console_log.info("Resuming — reloaded %d completed model(s): %s",
                         len(completed), list(completed.keys()))

    all_kpis = {}
    for model_name, run_data in completed.items():
        all_kpis[model_name] = run_data.get("kpis", {})

    # ── Model loop ────────────────────────────────────────────────────────────
    completed_names = list(completed.keys())
    sprint_counter = 0

    for model_name in MODEL_ORDER:
        mc = cfg["models"].get(model_name, {})
        if not mc.get("enabled", False):
            console_log.info("Skipping %s (disabled in params).", model_name)
            continue
        if model_name in completed:
            console_log.info("Skipping %s (checkpoint found, reloading).", model_name)
            continue

        # Per-model logger
        model_log, log_path = make_model_logger(exp_dir, model_name, cfg, run_id)
        console_log.info("── %s → %s", model_name, log_path.name)
        section(model_log, f"Model: {model_name}")
        model_log.info("Experiment : %s", cfg["experiment"]["name"])
        model_log.info("Run ID     : %s", run_id)
        model_log.info("Horizon    : %d", horizon)
        model_log.info("Train obs  : %d", len(train))

        try:
            # Select correct training input
            if model_name == "var":
                res = MODEL_FN[model_name](multi_train, horizon, cfg, model_log)
                if res is None:
                    continue
                forecast = res.forecast[cfg["data"]["target_col"]] if hasattr(res.forecast, "columns") else res.forecast
            else:
                res = MODEL_FN[model_name](train, horizon, cfg, model_log)
                forecast = res.forecast

            forecast.index = test.index

            # KPIs
            import time as _time
            _t_kpi = _time.perf_counter()
            kpis = compute_kpis(test, forecast)
            kpi_ms = (_time.perf_counter() - _t_kpi) * 1000
            kpis["model"] = model_name

            model_log.info("")
            model_log.info("── Test-set KPIs ─────────────────────────")
            model_log.info("  RMSE     : %.4f", kpis.get("RMSE", float("nan")))
            model_log.info("  MAE      : %.4f", kpis.get("MAE", float("nan")))
            model_log.info("  MAPE     : %.4f %%", kpis.get("MAPE", float("nan")))
            model_log.info("  R²       : %.4f", kpis.get("R2", float("nan")))
            model_log.info("  Coverage : %.4f", kpis.get("Coverage", float("nan")))
            model_log.info("  KPI calc : %.1f ms", kpi_ms)
            model_log.info("── Per-step forecast vs actual ───────────")
            for ts_idx, act, fc_val in zip(test.index, test.values, forecast.values):
                err = fc_val - act
                model_log.info("  %s  actual=%10.3f  forecast=%10.3f  error=%+.3f",
                               str(ts_idx)[:10], act, fc_val, err)

            # Persist artefacts
            save_run(exp_dir, model_name, res, kpis)
            save_test_run(exp_dir, model_name, test, forecast)

            # Forecast plot
            if cfg.get("output", {}).get("plots", False):
                import matplotlib
                matplotlib.use("Agg")
                from aic_ts_suite.viz import plot_forecast
                fig = plot_forecast(
                    observed=ts,
                    forecast=forecast,
                    lower=getattr(res, "lower", None),
                    upper=getattr(res, "upper", None),
                    train_end=train.index[-1],
                    kpis={k: v for k, v in kpis.items()
                          if k in ("RMSE", "MAE", "MAPE", "R2")},
                    title=f"{model_name.upper()} — Observed vs Forecast",
                )
                _save_fig(fig, exp_dir, f"{model_name}_forecast", model_log)

            # Checkpoint
            if ckpt_cfg.get("enabled", True):
                registry = save_checkpoint(exp_dir, model_name, res, kpis, keep_best_n, metric)
                model_log.info("Checkpoint saved | top-%d registry: %s",
                               keep_best_n,
                               [(e["model"], f"{e['score']:.4f}") for e in registry])
                console_log.info("  RMSE=%.4f | checkpoint saved | top-%d: %s",
                                 kpis.get("RMSE", float("nan")), keep_best_n,
                                 [e["model"] for e in registry])

            all_kpis[model_name] = kpis
            completed_names.append(model_name)

            # Sprint snapshot
            sprint_counter += 1
            if sprint_counter % sprint_interval == 0:
                sprint_num = sprint_counter // sprint_interval
                sprint_checkpoint(exp_dir, completed_names[:], sprint_num)
                model_log.info("Sprint %d checkpoint saved.", sprint_num)
                console_log.info("  Sprint %d checkpoint saved.", sprint_num)

        except Exception as exc:
            model_log.error("FAILED: %s", exc, exc_info=True)
            console_log.warning("  %s FAILED: %s", model_name, exc)

        finally:
            for h in model_log.handlers:
                h.close()
            model_log.handlers.clear()

    # ── Leaderboard ───────────────────────────────────────────────────────────
    section(console_log, "Leaderboard")
    rows = []
    for rank, (name, kpis) in enumerate(
        sorted(all_kpis.items(), key=lambda x: x[1].get("RMSE", float("inf"))), 1
    ):
        rows.append({
            "rank": rank,
            "model": name,
            "RMSE": round(kpis.get("RMSE", float("nan")), 4),
            "MAE": round(kpis.get("MAE", float("nan")), 4),
            "MAPE": round(kpis.get("MAPE", float("nan")), 4),
            "R2": round(kpis.get("R2", float("nan")), 4),
        })
    leaderboard = rows

    # Console table
    header = f"{'Rank':>4}  {'Model':<15}  {'RMSE':>10}  {'MAE':>10}  {'MAPE':>8}  {'R2':>8}"
    console_log.info(header)
    console_log.info("-" * len(header))
    for r in leaderboard:
        console_log.info("%4d  %-15s  %10.4f  %10.4f  %8.4f  %8.4f",
                         r["rank"], r["model"], r["RMSE"], r["MAE"], r["MAPE"], r["R2"])

    # Print unconditionally (leaderboard is always shown)
    print("\n" + header)
    print("-" * len(header))
    for r in leaderboard:
        print(f"{r['rank']:4d}  {r['model']:<15}  {r['RMSE']:10.4f}  "
              f"{r['MAE']:10.4f}  {r['MAPE']:8.4f}  {r['R2']:8.4f}")

    # ── Export ────────────────────────────────────────────────────────────────
    if cfg.get("output", {}).get("export_json", True):
        lb_path = exp_dir / "results" / "leaderboard.json"
        with lb_path.open("w") as f:
            json.dump(leaderboard, f, indent=2)

        registry = load_checkpoint_registry(exp_dir)
        summary = {
            "experiment": cfg["experiment"]["name"],
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "params_file": str(exp_dir / "params.yaml"),
            "best_model": leaderboard[0]["model"] if leaderboard else None,
            "top_3_checkpoints": registry,
            "models_run": completed_names,
            "leaderboard": leaderboard,
        }
        sum_path = exp_dir / "results" / "summary.json"
        with sum_path.open("w") as f:
            json.dump(summary, f, indent=2, default=str)

        console_log.info("Results written to %s", exp_dir / "results")
        print(f"\nResults → {exp_dir / 'results'}")
        print(f"Logs    → {exp_dir / 'logs'}")

    # ── Comparison plot (all models on one axes) ──────────────────────────────
    if cfg.get("output", {}).get("plots", False) and all_kpis:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colours = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(ts.index, ts.values, label="Observed", color="#2c3e50",
                linewidth=1.0, alpha=0.7)
        ax.axvline(train.index[-1], linestyle=":", color="grey",
                   linewidth=0.9, label="Train / Test split")
        for (model_name, _kpis), colour in zip(
            sorted(all_kpis.items(), key=lambda x: x[1].get("RMSE", float("inf"))),
            colours,
        ):
            run_path = exp_dir / "runs" / f"{model_name}_run.json"
            if not run_path.exists():
                continue
            with run_path.open() as f:
                run_data = json.load(f)
            fc_vals = run_data.get("forecast") or []
            if not fc_vals:
                continue
            fc_series = pd.Series(fc_vals, index=test.index)
            rmse_val = _kpis.get("RMSE", float("nan"))
            ax.plot(fc_series.index, fc_series.values,
                    label=f"{model_name} (RMSE={rmse_val:.1f})",
                    color=colour, linewidth=1.4, linestyle="--")
        ax.set_title("All Models — Forecast Comparison")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save_fig(fig, exp_dir, "leaderboard_comparison", console_log)
        print(f"Plots   → {exp_dir / 'plots'}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    """Parse args, validate config, set up experiment dir, run pipeline."""
    args = parse_args(argv)

    cfg = load_params(args.params)
    cfg["_source_path"] = args.params  # stash for copying into exp_dir

    console_log = make_console_logger(args.verbose, cfg)

    exp_dir = make_experiment_dir(cfg, args.resume)
    console_log.info("Experiment dir: %s", exp_dir)
    print(f"\nExperiment → {exp_dir}\n")

    run_experiment(cfg, exp_dir, console_log)


if __name__ == "__main__":
    main()
