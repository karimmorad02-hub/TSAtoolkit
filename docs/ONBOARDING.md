# Onboarding Guide — aic_ts_suite

Welcome to **aic_ts_suite**, the Analytics Engineering team's modular time-series forecasting toolkit. This guide will get you from zero to running experiments in under an hour.

---

## Table of Contents

1. [What Is This Toolkit?](#what-is-this-toolkit)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Core Concepts](#core-concepts)
6. [Your First Experiment](#your-first-experiment)
7. [Module Reference](#module-reference)
   - [Configuration](#1-configuration)
   - [Data Ingestion](#2-data-ingestion)
   - [Data Cleaning](#3-data-cleaning)
   - [Signal Transforms](#4-signal-transforms)
   - [Feature Engineering](#5-feature-engineering)
   - [Forecasting Models](#6-forecasting-models)
   - [Evaluation & Comparison](#7-evaluation--comparison)
   - [Visualization](#8-visualization)
   - [Traceability](#9-traceability)
8. [Running Experiments at Scale](#running-experiments-at-scale)
   - [The Experiment Runner](#the-experiment-runner)
   - [Configuration via params.yaml](#configuration-via-paramsyaml)
   - [Checkpointing & Resumption](#checkpointing--resumption)
9. [Deployment](#deployment)
10. [Weather Data Integration](#weather-data-integration)
11. [Troubleshooting](#troubleshooting)
12. [Glossary](#glossary)

---

## What Is This Toolkit?

aic_ts_suite is an end-to-end time-series forecasting framework built for R&D workflows. It covers the full pipeline:

```
Raw Data → Cleaning → Feature Engineering → Model Training → Evaluation → Visualization
```

**Design principles:**

- **Fluent API** — method chaining for clean, readable pipelines
- **Modular** — each sub-package handles one concern; use only what you need
- **Model-agnostic comparison** — all forecasters produce a standard `ForecastResult`, making side-by-side evaluation trivial
- **Traceability** — every run gets a UUID (`correlationId`) propagated through all logs
- **Checkpoint-driven** — the experiment runner saves progress so you can resume interrupted runs

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | 3.11 recommended |
| pip | latest | `pip install --upgrade pip` |
| Git | any | for cloning |
| Jupyter | optional | for notebook workflows |

**Optional (for specific features):**

| Feature | Requirement |
|---------|-------------|
| TimescaleDB connector | PostgreSQL + psycopg2 |
| Neural models (NHITS, MLP) | PyTorch + neuralforecast |
| TimeGPT | nixtla package + `NIXTLA_API_KEY` env var |
| Kubernetes deployment | kubectl + cluster access |

---

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd TSAtoolkit

# Core install (statistical models + viz)
pip install -e .

# With ML models (XGBoost + Prophet)
pip install -e ".[ml]"

# Everything (neural + foundation models)
pip install -e ".[all]"
```

Verify the install:

```bash
python -c "from aic_ts_suite.config import CONFIG; print(f'Ready — correlation_id: {CONFIG.correlation_id}')"
```

---

## Project Structure

```
TSAtoolkit/
├── aic_ts_suite/                  # Main package
│   ├── config.py                  # Global CONFIG singleton
│   ├── connectivity/              # Data ingestion (DB, CSV, Excel, weather API)
│   ├── cleaning/                  # Missing-value handling, DataCleaner pipeline
│   ├── signals/                   # Variance-stabilising transforms (Log, Sqrt, Box-Cox)
│   ├── features/                  # Lags, rolling stats, Fourier harmonics, moving averages
│   ├── forecasting/               # All model families (statistical, ML, neural, foundation)
│   ├── evaluation/                # KPI metrics, model comparison, run logging
│   ├── viz/                       # Plots (forecast, seasonal, ACF/PACF, decomposition)
│   └── display/                   # Rich notebook rendering helpers
├── configs/
│   └── params.yaml                # Experiment parameters (YAML)
├── scripts/
│   └── toolkit_demo.py            # Automated experiment runner with checkpointing
├── notebooks/
│   └── toolkit_demo.ipynb         # Interactive demo notebook
├── prompts/
│   └── experiment_prompt.md       # MLOps orchestration prompt
├── kubernetes/
│   └── deployment.yaml            # K8s Job spec
├── requirements.txt
├── setup.py
└── docker-compose.yaml
```

---

## Core Concepts

### ForecastResult

Every forecaster returns a `ForecastResult` — the universal output format:

```python
@dataclass
class ForecastResult:
    model_name: str                  # e.g. "AutoARIMA"
    forecast: pd.Series              # Point forecasts (datetime-indexed)
    lower: pd.Series | None          # Lower prediction interval bound
    upper: pd.Series | None          # Upper prediction interval bound
    fitted_values: pd.Series | None  # In-sample fitted values
    duration_ms: float               # Wall-clock fit+predict time
    info_criteria: dict              # AIC, BIC, AICc (where applicable)
    correlation_id: str              # UUID for traceability
    extra: dict                      # Model-specific metadata
```

This standardisation means you can compare any model against any other — ARIMA vs XGBoost vs Prophet — through the same `ModelComparison` interface.

### Correlation ID

Every run is tagged with a UUID v4 via `CONFIG.correlation_id`. This ID flows through forecasters, evaluation, and logging, enabling end-to-end traceability across system logs.

```python
from aic_ts_suite.config import CONFIG

print(CONFIG.correlation_id)       # e.g. "a3f2c1d4-..."
CONFIG.refresh_correlation_id()    # Generate a new ID for the next run
```

### Fluent API

The `DataCleaner` uses method chaining so you can express an entire cleaning pipeline in one statement:

```python
df = (DataCleaner("data.csv", timestamp_col="date")
      .load()
      .sanitize(strategy="interpolate")
      .to_epoch_ms()
      .set_datetime_index()
      .result())
```

---

## Your First Experiment

This walkthrough takes you from raw CSV to a model leaderboard in ~20 lines.

```python
from aic_ts_suite.config import CONFIG
from aic_ts_suite.cleaning import DataCleaner
from aic_ts_suite.forecasting import (
    auto_select_univariate, XGBoostForecaster, ProphetForecaster
)
from aic_ts_suite.evaluation import ModelComparison
from aic_ts_suite.viz import plot_forecast, plot_decomposition

# 1. Load & clean
df = (DataCleaner("Data/time_series_15min.csv", timestamp_col="utc_timestamp")
      .load()
      .sanitize()
      .to_epoch_ms()
      .set_datetime_index()
      .result())

target = df["AT_load_actual_entsoe_transparency"]

# 2. Train/test split
train, test = target.iloc[:-12], target.iloc[-12:]

# 3. EDA (optional but recommended)
plot_decomposition(train, period=12)

# 4. Fit multiple models
cmp = ModelComparison(test)

cmp.add(auto_select_univariate(train, horizon=12, seasonal_periods=12))
cmp.add(XGBoostForecaster(lags=range(1, 13), rolling_windows=[3, 6, 12]).fit_predict(train, 12))
cmp.add(ProphetForecaster(yearly_seasonality=True).fit_predict(train, 12))

# 5. Compare
cmp.leaderboard(sort_by="RMSE")

# 6. Visualise the best model
best = cmp.leaderboard().iloc[0]
plot_forecast(observed=target, forecast=best.forecast, train_end=train.index[-1])
```

---

## Module Reference

### 1. Configuration

```python
from aic_ts_suite.config import CONFIG
```

| Attribute | Default | Description |
|-----------|---------|-------------|
| `correlation_id` | auto-generated UUID v4 | Unique run identifier |
| `confidence_level` | `0.95` | Prediction interval width |
| `random_state` | `42` | Reproducibility seed |
| `default_sanitize_strategy` | `"interpolate"` | Missing-value strategy |
| `timescale.host` | `TSDB_HOST` env / `"localhost"` | TimescaleDB host |
| `timescale.port` | `TSDB_PORT` env / `5432` | TimescaleDB port |
| `timescale.database` | `TSDB_DATABASE` env / `"tsdb"` | Database name |
| `timescale.user` | `TSDB_USER` env / `"readonly"` | DB username |
| `timescale.password` | `TSDB_PASSWORD` env | DB password |
| `timescale.schema` | `TSDB_SCHEMA` env / `"public"` | Schema |

---

### 2. Data Ingestion

#### CSV / Excel

```python
from aic_ts_suite.connectivity.file_io import read_csv, read_excel, scan_directory

df = read_csv("data.csv", timestamp_col="date", value_cols=["col1", "col2"])
df = read_excel("data.xlsx", sheet_name=0, timestamp_col="date")
files = scan_directory("data/", recursive=True)
```

#### TimescaleDB

```python
from aic_ts_suite.connectivity.timescale import TimescaleClient

client = TimescaleClient()  # uses CONFIG.timescale
df = client.query("SELECT time, value FROM metrics WHERE sensor_id = %(sid)s",
                  params={"sid": "sensor_123"}, parse_dates=["time"])
client.close()
```

#### Canonical Data Model

All data is internally represented as `TimeSeriesObservation`:

```python
from aic_ts_suite.connectivity.models import TimeSeriesObservation

obs = TimeSeriesObservation(timestamp_ms=1609459200000, value=42.5, series_id="load")
print(obs.datetime_utc)  # timezone-aware datetime
```

---

### 3. Data Cleaning

```python
from aic_ts_suite.cleaning import DataCleaner
from aic_ts_suite.cleaning.sanitize import sanitize

# Fluent pipeline
df = DataCleaner("data.csv", timestamp_col="date").load().sanitize().to_epoch_ms().set_datetime_index().result()

# Standalone sanitize
df_clean = sanitize(df, strategy="interpolate", value_cols=["col1"])
```

**Strategies:**

| Strategy | Behaviour |
|----------|-----------|
| `"interpolate"` | Linear interpolation + both-ends fill |
| `"ffill"` | Forward-fill + backward-fill for leading NaNs |

---

### 4. Signal Transforms

Variance-stabilising transforms with automatic inverse:

```python
from aic_ts_suite.signals.transforms import LogTransform, SqrtTransform, BoxCoxTransform

xform = BoxCoxTransform()
transformed = xform.apply(series)     # auto-estimates lambda via MLE
original    = xform.inverse(transformed)
print(xform.params)                   # {"lmbda": ..., "offset": ...}
```

All transforms handle non-positive data automatically via offset.

---

### 5. Feature Engineering

#### Lag Features

```python
from aic_ts_suite.features.lags import lag_features, rolling_lag_features, build_supervised_matrix

# Individual lags
lags_df = lag_features(ts, lags=[1, 2, 3, 6, 12])

# Rolling statistics
roll_df = rolling_lag_features(ts, windows=[3, 6, 12], stats=["mean", "std"])

# All-in-one supervised matrix (lags + rolling + Fourier)
X = build_supervised_matrix(ts, lags=range(1, 13), rolling_windows=[3, 6, 12],
                            fourier_k=3, fourier_period=12.0, drop_na=True)
```

#### Fourier Harmonics

```python
from aic_ts_suite.features.fourier import fourier_terms, optimal_k

K = optimal_k(ts, period=12.0, max_K=6)   # AICc-based selection
ft = fourier_terms(n=len(ts), period=12.0, K=K)
# Columns: sin_1, cos_1, sin_2, cos_2, ...
```

#### Moving Averages

```python
from aic_ts_suite.features.moving_averages import trailing_moving_average, centered_moving_average

trail  = trailing_moving_average(ts, window=12)   # safe for prediction (backward-looking)
center = centered_moving_average(ts, window=12)    # symmetric (offline analysis only)
```

---

### 6. Forecasting Models

All forecasters follow the same interface:

```python
model = SomeForecaster(**params)
result = model.fit_predict(train, horizon=12)  # returns ForecastResult
```

#### Statistical (Univariate)

| Model | Class | Key Parameters |
|-------|-------|----------------|
| AutoARIMA | `AutoARIMAForecaster` | `seasonal`, `m` |
| AutoETS | `AutoETSForecaster` | `seasonal_periods` |
| Holt-Winters | `HoltWintersForecaster` | `seasonal` ("add"/"mul"), `seasonal_periods` |
| Auto-select | `auto_select_univariate()` | Runs all three, returns best by AICc |

```python
from aic_ts_suite.forecasting import AutoARIMAForecaster, auto_select_univariate

result = AutoARIMAForecaster(seasonal=True, m=12).fit_predict(train, horizon=12)
best   = auto_select_univariate(train, horizon=12, seasonal_periods=12)
```

#### Machine Learning

| Model | Class | Key Parameters |
|-------|-------|----------------|
| XGBoost | `XGBoostForecaster` | `lags`, `rolling_windows`, `fourier_k`, `n_estimators`, `learning_rate`, `max_depth` |
| Prophet | `ProphetForecaster` | `growth`, `seasonality_mode`, `yearly/weekly/daily_seasonality`, `changepoint_prior_scale` |

```python
from aic_ts_suite.forecasting import XGBoostForecaster, ProphetForecaster

xgb = XGBoostForecaster(lags=range(1, 13), rolling_windows=[3, 6, 12],
                         fourier_k=3, n_estimators=300, learning_rate=0.05, max_depth=5)
result = xgb.fit_predict(train, horizon=12)
xgb.feature_importance().head(10)  # inspect what matters

prophet = ProphetForecaster(growth="linear", seasonality_mode="additive",
                             yearly_seasonality=True, changepoint_prior_scale=0.05)
result = prophet.fit_predict(train, horizon=12)
```

**How XGBoost works internally:**
1. `build_supervised_matrix()` creates features from lags, rolling stats, and Fourier harmonics
2. Three `XGBRegressor` models are fitted — point estimate + quantile upper/lower for prediction intervals
3. Multi-step forecasting is recursive: each prediction feeds back as input for the next step

#### Multivariate (VAR)

```python
from aic_ts_suite.forecasting import VARForecaster

var = VARForecaster(maxlags=12, ic="aic", granger_maxlag=12, granger_alpha=0.05)
result = var.fit_predict(multi_df, horizon=12)
# Granger causality results: result.extra["granger"]
```

#### Neural & Foundation Models

```python
from aic_ts_suite.forecasting import NHITSForecaster, MLPForecaster, TimeGPTForecaster

# NHITS / MLP (requires neuralforecast + PyTorch)
result = NHITSForecaster(horizon=12, max_steps=200).fit_predict(train, 12)
result = MLPForecaster(horizon=12, max_steps=200).fit_predict(train, 12)

# TimeGPT — zero-shot, no training (requires nixtla + NIXTLA_API_KEY env var)
result = TimeGPTForecaster(horizon=12).fit_predict(train, 12)
```

---

### 7. Evaluation & Comparison

#### Metrics

```python
from aic_ts_suite.evaluation import mae, rmse, mape, smape, r_squared, compute_all_kpis

kpis = compute_all_kpis(observed, forecast)
# {"MAE": ..., "RMSE": ..., "MAPE": ..., "sMAPE": ..., "R²": ...}
```

| Metric | What It Measures | When to Use |
|--------|-----------------|-------------|
| **MAE** | Average absolute error | General-purpose, interpretable in original units |
| **RMSE** | Root mean square error | When large errors are especially costly |
| **MAPE** | Percentage error | Cross-series comparison (avoid when values near zero) |
| **sMAPE** | Symmetric percentage error | Robust alternative to MAPE for low-value data |
| **R²** | Variance explained | Overall model quality (1.0 = perfect) |

#### Model Comparison

```python
from aic_ts_suite.evaluation import ModelComparison

cmp = ModelComparison(test)
cmp.add(result_arima)
cmp.add(result_xgb)
cmp.add(result_prophet)

cmp.leaderboard(sort_by="RMSE", show=True)        # styled HTML table in notebooks
cmp.metric_deltas(baseline_model="AutoARIMA")      # relative improvement/regression
cmp.plot_all(observed_full, train_end=split_date)   # overlay all forecasts
```

---

### 8. Visualization

All plot functions return a matplotlib `Figure` and auto-display in Jupyter.

| Function | Purpose |
|----------|---------|
| `plot_forecast(observed, forecast, lower, upper, train_end, kpis)` | Observed vs forecast with prediction interval + KPI annotation box |
| `plot_seasonal(series, period, freq_label)` | Year-over-year seasonal overlay with colour gradients |
| `plot_acf_pacf(series, lags, alpha)` | Side-by-side ACF/PACF with 95% confidence bands |
| `plot_decomposition(series, period, model)` | 4-panel decomposition: Observed, Trend, Seasonal, Residual |
| `styled_summary(df, caption)` | CSS-styled HTML table |
| `metric_cards(kpis_dict)` | KPI metric cards for dashboard-style display |

```python
from aic_ts_suite.viz import (
    plot_forecast, plot_seasonal, plot_acf_pacf, plot_decomposition,
    styled_summary, metric_cards
)

plot_decomposition(train, period=12, model="additive")
plot_acf_pacf(train, lags=40)
plot_forecast(observed=target, forecast=result.forecast,
              lower=result.lower, upper=result.upper,
              train_end=train.index[-1],
              kpis={"RMSE": 12.34, "R²": 0.85})
```

---

### 9. Traceability

Log every run for audit and pipeline integration:

```python
from aic_ts_suite.evaluation import AnalyticsEngineClient

client = AnalyticsEngineClient()
client.log_run(result, test)           # records model, KPIs, correlation_id
client.summary()                       # DataFrame of all logged runs
client.to_json()                       # JSON export for downstream systems
```

---

## Running Experiments at Scale

### The Experiment Runner

The script `scripts/toolkit_demo.py` automates multi-model experiments with full checkpointing:

```bash
# Fresh run
python scripts/toolkit_demo.py --params configs/params.yaml

# Verbose output
python scripts/toolkit_demo.py --params configs/params.yaml --verbose

# Resume an interrupted run
python scripts/toolkit_demo.py --params configs/params.yaml \
    --resume experiments/energy_forecast_20260324_143200_a3f2
```

**What it does:**

1. Validates and loads `params.yaml`
2. Creates an experiment directory: `experiments/{name}_{timestamp}_{run_id}/`
3. Iterates through enabled models in the config
4. Saves checkpoints (top N by metric)
5. Produces per-model logs, plots, and run JSONs
6. Exports a final leaderboard

### Output Structure

```
experiments/{name}_{timestamp}_{run_id}/
├── params.yaml              # Frozen copy of config
├── checkpoints/
│   ├── best_checkpoints.json
│   ├── checkpoint_arima_312.45_20260324.pkl
│   └── sprint_001.json
├── runs/                    # Per-model run records
├── test_runs/               # Test-set evaluations
├── logs/                    # Searchable per-model logs
├── plots/                   # Forecast PNGs
└── results/
    ├── leaderboard.json
    └── summary.json
```

### Configuration via params.yaml

The experiment is fully controlled by `configs/params.yaml`. Key sections:

```yaml
experiment:
  name: "energy_forecast"
  tags: [energy, austria]

data:
  csv_path: "Data/time_series_15min.csv"
  timestamp_col: "utc_timestamp"
  target_col: "AT_load_actual_entsoe_transparency"
  max_rows: 5000              # null = use all rows

weather:
  enabled: true
  latitude: 47.8095
  longitude: 13.0550
  variables: [temperature_2m, precipitation, wind_speed_10m]

modelling:
  horizon: 12
  seasonal_period: 12
  freq: "MS"

models:
  arima:
    enabled: true
    seasonal: true
  xgboost:
    enabled: true
    n_estimators: 300
    learning_rate: 0.05
    max_depth: 5
    lags: 12
    rolling_windows: [3, 6, 12]
    fourier_k: 4
  prophet:
    enabled: true
    growth: "linear"
  # ... (ets, holt_winters, var also available)

checkpoints:
  enabled: true
  keep_best_n: 3
  metric: "RMSE"
```

### Checkpointing & Resumption

- Checkpoints are saved after each model completes
- The top N checkpoints (by metric) are retained; older ones are pruned
- Sprint snapshots are flushed every N models (configurable)
- To resume, pass `--resume <experiment_dir>` — already-completed models are loaded from checkpoint and skipped

---

## Deployment

### Kubernetes

The `kubernetes/deployment.yaml` defines a K8s Job configured for GPU clusters. To adapt for your environment:

1. Update `namespace`, `queue-name`, and container `image`
2. Adjust resource requests/limits (default: 5 CPU, 50 GB RAM)
3. Bind the correct PVC for your NFS/storage mount
4. Apply: `kubectl apply -f kubernetes/deployment.yaml`

### Docker

`docker-compose.yaml` is a placeholder. A minimal setup for local development with TimescaleDB:

```yaml
version: "3.8"
services:
  timescale:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
```

Set `TSDB_HOST=localhost` and `TSDB_PASSWORD=password` in your environment to connect.

---

## Weather Data Integration

The toolkit integrates free weather data from the Open-Meteo API (no API key required):

```python
from aic_ts_suite.connectivity.weather import fetch_weather, resample_weather, merge_weather

# Fetch historical hourly weather
weather = fetch_weather(
    latitude=47.8095, longitude=13.0550,
    variables=["temperature_2m", "precipitation", "wind_speed_10m"],
    start_date="2020-01-01", end_date="2023-12-31",
    source="archive"
)

# Resample to match your main data frequency
weather_daily = resample_weather(weather, freq="D")

# Merge into your main DataFrame
merged = merge_weather(main_df, weather, freq="D", how="left", prefix="weather_")
```

**Available sources:**

| Source | Use Case |
|--------|----------|
| `"archive"` | Historical data (past dates) |
| `"forecast"` | Future weather forecasts |
| `"auto"` | Automatically selects based on date range |

Weather features (temperature, precipitation, wind, cloud cover, etc.) can significantly improve energy and demand forecasting models.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `MAPE` warns about near-zero values | Use `sMAPE` instead — it's symmetric and robust |
| XGBoost runs out of memory | Reduce `n_estimators` or `max_depth` in config |
| Prophet is very slow | Subsample data or shorten the training period |
| TimescaleDB connection fails | Verify `TSDB_*` environment variables; inspect `CONFIG.timescale.dsn` |
| Weather API timeout | Increase `timeout` in params.yaml or use a smaller date range |
| Notebook kernel crashes | Set `max_rows` in params.yaml to limit data size |
| Neural model import errors | Install optional deps: `pip install -e ".[neural]"` or `pip install -e ".[all]"` |
| TimeGPT authentication | Set the `NIXTLA_API_KEY` environment variable |

---

## Glossary

| Term | Definition |
|------|------------|
| **Correlation ID** | UUID v4 assigned per run for end-to-end log traceability |
| **ForecastResult** | Standardised model output: forecast + prediction intervals + metadata |
| **Lag feature** | Shifted copy of the series (e.g. lag_1 = y[t-1]); captures autoregressive structure |
| **Fourier term** | Sin/cos pair at a harmonic frequency; models seasonal patterns without dummy variables |
| **Prediction Interval (PI)** | Probabilistic bounds (e.g. 95%) around a point forecast |
| **AICc** | Corrected Akaike Information Criterion; used for automatic model selection |
| **Granger causality** | Statistical test for whether one time series helps predict another |
| **Horizon** | Number of time steps to forecast into the future |
| **Seasonal period** | Length of one seasonal cycle (e.g. 12 for monthly data with yearly seasonality) |
| **Checkpoint** | Serialised model state + results; enables experiment resumption and best-model tracking |
| **sMAPE** | Symmetric Mean Absolute Percentage Error; robust to near-zero actuals |
| **Supervised matrix** | Feature table (lags + rolling stats + Fourier) with target column, ready for ML training |

---

## Next Steps

1. **Run the demo notebook** — `jupyter notebook notebooks/toolkit_demo.ipynb`
2. **Run an automated experiment** — `python scripts/toolkit_demo.py --params configs/params.yaml --verbose`
3. **Customise `params.yaml`** — point to your data, toggle models on/off, tune hyperparameters
4. **Add weather features** — enable the weather section in params.yaml for energy/demand data
5. **Explore multivariate** — use `VARForecaster` with multiple related series for cross-variable forecasting
6. **Deploy** — adapt `kubernetes/deployment.yaml` for your cluster

Questions? Reach out to the Analytics Engineering team.
