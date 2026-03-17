# aic_ts_suite — Modular Time-Series Notebook Toolkit

An interactive R&D extension bridging raw data ingestion and advanced predictive modelling. Designed for Jupyter / IPython environments.

---

## Architecture

```
aic_ts_suite/
├── config.py                    # Global config, correlationId, TimescaleDB params
├── connectivity/
│   ├── models.py                # TimeSeriesObservation data model
│   ├── timescale.py             # TimescaleDB connector (SQLAlchemy)
│   └── file_io.py               # CSV / Excel (.xlsx, .xls) ingestion
├── cleaning/
│   ├── sanitize.py              # Missing-value strategies (interpolate, ffill)
│   └── data_cleaner.py          # Fluent DataCleaner pipeline
├── signals/
│   └── transforms.py            # Log, Sqrt, Box-Cox (λ via MLE)
├── viz/
│   ├── seasonal.py              # Seasonal overlay with colour gradients
│   ├── acf_pacf.py              # Side-by-side ACF / PACF with 95% CI bands
│   ├── decomposition.py         # 4-panel Y=T+S+R with reconstruction error
│   ├── forecast_plot.py         # Observed vs Forecast + PI + KPI annotation
│   └── styles.py                # CSS/HTML styled tables & metric cards
├── features/
│   ├── fourier.py               # fourier_terms(), optimal_k() (AICc)
│   ├── moving_averages.py       # Trailing & Centered MAs
│   └── lags.py                  # Autoregressive lags, rolling stats, supervised matrix
├── forecasting/
│   ├── base.py                  # BaseForecaster, ForecastResult
│   ├── univariate.py            # AutoARIMA, AutoETS, Holt-Winters, auto_select
│   ├── multivariate.py          # VAR + Granger causality
│   ├── ml_models.py             # XGBoost (lag-based) & Prophet
│   └── neural.py                # NHITS, MLP (NeuralForecast), TimeGPT (Nixtla)
├── evaluation/
│   ├── metrics.py               # MAE, RMSE, MAPE, sMAPE, R²
│   ├── comparison.py            # ModelComparison leaderboard + delta tracking
│   └── engine_client.py         # AnalyticsEngineClient (run traceability)
└── display/
    └── __init__.py              # Re-exports styled_summary, metric_cards
```

---

## Quick Start

```bash
pip install -e .
# with ML models (XGBoost + Prophet):
pip install -e ".[ml]"
# all optional dependencies:
pip install -e ".[all]"
```

```python
from aic_ts_suite.config import CONFIG
from aic_ts_suite.cleaning import DataCleaner
from aic_ts_suite.forecasting import auto_select_univariate
from aic_ts_suite.evaluation import ModelComparison

# Clean
df = DataCleaner("data.csv").load().sanitize().to_epoch_ms().result()

# Forecast
ts = df.set_index("timestamp")["value"]
train, test = ts.iloc[:-12], ts.iloc[-12:]
result = auto_select_univariate(train, horizon=12)

# Evaluate
cmp = ModelComparison(test)
cmp.add(result)
cmp.leaderboard()
```

---

## Feature Engineering

### Lag Features (`features.lags`)

Autoregressive lag columns and rolling statistics for supervised ML models.

```python
from aic_ts_suite.features import lag_features, rolling_lag_features, build_supervised_matrix

# Individual lag columns
lags_df = lag_features(ts, lags=[1, 2, 3, 6, 12])

# Rolling mean / std
roll_df = rolling_lag_features(ts, windows=[3, 6, 12], stats=["mean", "std"])

# Complete supervised matrix (lags + rolling stats + optional Fourier)
X = build_supervised_matrix(
    ts,
    lags=range(1, 13),
    rolling_windows=[3, 6, 12],
    fourier_K=3,
    fourier_period=12.0,
)
```

| Function | Description |
|----------|-------------|
| `lag_features(series, lags)` | Creates `lag_1`, `lag_2`, … columns by shifting the series |
| `rolling_lag_features(series, windows, stats)` | Rolling aggregates (`roll_mean_3`, `roll_std_6`, …) |
| `build_supervised_matrix(series, …)` | All-in-one: lags + rolling + Fourier → ready-to-train `DataFrame` |

### Fourier Harmonics (`features.fourier`)

```python
from aic_ts_suite.features import fourier_terms, optimal_k

K = optimal_k(ts, period=12)           # AICc-based harmonic selection
ft = fourier_terms(n=len(ts), period=12, K=K)
```

### Moving Averages (`features.moving_averages`)

```python
from aic_ts_suite.features import trailing_moving_average, centered_moving_average

trail  = trailing_moving_average(ts, window=12)  # backward-looking (safe for prediction)
center = centered_moving_average(ts, window=12)   # symmetric (offline analysis only)
```

---

## Forecasting Algorithms

### Univariate (AutoARIMA, AutoETS, Holt-Winters)

```python
from aic_ts_suite.forecasting import AutoARIMAForecaster, auto_select_univariate

result = AutoARIMAForecaster(seasonal=True, m=12).fit_predict(train, horizon=12)

# Or auto-select the best by AICc:
best = auto_select_univariate(train, horizon=12, seasonal_periods=12)
```

### XGBoost (`forecasting.ml_models.XGBoostForecaster`)

Gradient-boosted tree forecaster that builds a supervised lag/rolling/Fourier
feature matrix and uses recursive multi-step prediction.  Prediction intervals
are estimated via XGBoost quantile regression.

```python
from aic_ts_suite.forecasting import XGBoostForecaster

xgb = XGBoostForecaster(
    lags=range(1, 13),          # 12 autoregressive lags
    rolling_windows=[3, 6, 12], # rolling mean & std features
    fourier_K=3,                # 3 harmonic sin/cos pairs
    fourier_period=12.0,
)
result = xgb.fit_predict(train, horizon=12)

# Inspect feature importances
xgb.feature_importance().head(10)
```

**How it works:**

1. `build_supervised_matrix()` creates the feature matrix from lags, rolling
   statistics, and Fourier harmonics.
2. Three `XGBRegressor` models are fitted — one for point estimation, two for
   quantile prediction intervals (lower/upper bounds).
3. Multi-step forecasting is performed recursively: each prediction is appended
   to the history so subsequent lag features remain valid.

### Prophet (`forecasting.ml_models.ProphetForecaster`)

Meta/Facebook Prophet with additive or multiplicative seasonality, built-in
uncertainty intervals, and changepoint detection.

```python
from aic_ts_suite.forecasting import ProphetForecaster

prophet = ProphetForecaster(
    growth="linear",
    seasonality_mode="additive",
    yearly_seasonality=True,
    changepoint_prior_scale=0.05,
)
result = prophet.fit_predict(train, horizon=12)
```

### Multivariate — VAR

```python
from aic_ts_suite.forecasting import VARForecaster

var = VARForecaster(maxlags=12, ic="aic")
result = var.fit_predict(multi_df, horizon=12)
# Granger causality results in result.extra["granger"]
```

### Neural & Foundation Models

```python
from aic_ts_suite.forecasting import NHITSForecaster, MLPForecaster, TimeGPTForecaster

# NHITS / MLP (requires neuralforecast)
res_nhits = NHITSForecaster(horizon=12).fit_predict(train, 12)

# TimeGPT zero-shot (requires nixtla + API key)
res_tgpt = TimeGPTForecaster(horizon=12).fit_predict(train, 12)
```

---

## KPI Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **MAE** | mean(\|y − ŷ\|) | Average error in original units |
| **RMSE** | √mean((y − ŷ)²) | Penalises large outliers |
| **MAPE** | mean(\|y − ŷ\| / \|y\|) × 100 | Percentage-based reliability (warns near zero) |
| **sMAPE** | mean(\|y − ŷ\| / ((|y| + |ŷ|)/2)) × 100 | Robust percentage error for low-value data |
| **R²** | 1 − SS_res / SS_tot | Proportion of variance explained |

```python
from aic_ts_suite.evaluation import compute_all_kpis, ModelComparison

kpis = compute_all_kpis(test, result.forecast)

# Side-by-side leaderboard
cmp = ModelComparison(test)
cmp.add(res_arima)
cmp.add(res_xgb)
cmp.add(res_prophet)
cmp.leaderboard(sort_by="RMSE")
```

---

## Traceability

Every run propagates a `correlationId` (UUID v4) through `CONFIG.correlation_id`, logged by the `AnalyticsEngineClient` for end-to-end traceability across system logs.

```python
from aic_ts_suite.evaluation import AnalyticsEngineClient

client = AnalyticsEngineClient()
client.log_run(result, test)
print(client.to_json())
```

---

## Dependencies

| Category | Packages |
|----------|----------|
| **Core** | pandas, numpy, scipy, statsmodels, pmdarima, matplotlib |
| **ML** | xgboost, prophet |
| **Neural** | neuralforecast (NHITS/MLP) |
| **Foundation** | nixtla (TimeGPT) |
| **Database** | psycopg2-binary (TimescaleDB) |

---

## License

Internal R&D toolkit — Analytics Engineering Team.
