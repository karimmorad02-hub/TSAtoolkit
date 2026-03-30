# aic_ts_suite — CTO Technical Briefing

## Time-Series Forecasting Toolkit | Analytics Engineering

**Date:** 2026-03-29
**Status:** Production-Ready (v0.1.0)
**Python:** 3.10+ | **Codebase:** ~3,000 LOC across 35 modules

---

## 1. Executive Summary

**aic_ts_suite** is a modular, end-to-end time-series forecasting toolkit that unifies data ingestion, cleaning, feature engineering, model training, evaluation, and visualization into a single fluent API. It supports **9 forecasting algorithms** across 4 paradigms (classical, ML, neural, foundation models), evaluates them against **5 standardised KPIs**, and provides full **UUID-based run traceability** — from raw data to production leaderboard.

### Key Differentiators

| Capability | Value |
|-----------|-------|
| **Model-agnostic comparison** | Any model vs any model through a unified `ForecastResult` interface |
| **Zero-to-leaderboard in 20 lines** | Fluent API eliminates boilerplate |
| **YAML-driven experiment orchestration** | Reproducible, checkpoint-resumable runs |
| **End-to-end traceability** | UUID v4 correlation IDs across all modules and logs |
| **Modular by design** | Use only what you need — each sub-package is independent |

---

## 2. Architecture Overview

### Interactive Diagram

**[View Module Architecture on Excalidraw](https://excalidraw.com/#json=JRc6HIQ6JCYY6Hgb25AjN,SeNIBmzVCsEGjPgApUiVAg)**

### Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EVALUATION & OUTPUT                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  evaluation/  │  │    viz/      │  │      display/            │  │
│  │  5 KPIs      │  │  5 plot      │  │  Styled HTML tables      │  │
│  │  Leaderboard │  │  types       │  │  Metric cards            │  │
│  │  Traceability│  │              │  │                          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                       FORECASTING ENGINE                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  ┌─────────┐  │
│  │ univariate   │  │ ml_models    │  │ neural     │  │ multi-  │  │
│  │ AutoARIMA    │  │ XGBoost      │  │ NHITS      │  │ variate │  │
│  │ AutoETS      │  │ Prophet      │  │ MLP        │  │ VAR +   │  │
│  │ Holt-Winters │  │              │  │ TimeGPT    │  │ Granger │  │
│  └──────────────┘  └──────────────┘  └────────────┘  └─────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                     TRANSFORM & FEATURES                           │
│  ┌──────────────────────────┐  ┌────────────────────────────────┐  │
│  │  signals/                │  │  features/                     │  │
│  │  Log, Sqrt, Box-Cox (λ) │  │  Lags, Rolling Stats, Fourier  │  │
│  └──────────────────────────┘  └────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                       DATA INGESTION                               │
│  ┌───────────┐  ┌──────────────┐  ┌────────────┐  ┌────────────┐  │
│  │  config    │  │ connectivity │  │  cleaning   │  │  weather   │  │
│  │  Singleton │  │ CSV/Excel    │  │  DataCleaner│  │  Open-Meteo│  │
│  │  UUID      │  │ TimescaleDB  │  │  Fluent API │  │  API       │  │
│  └───────────┘  └──────────────┘  └────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow — End to End

### Interactive Diagram

**[View Data Pipeline Flow on Excalidraw](https://excalidraw.com/#json=I18_GDCG_l5pUvUtMCJi0,E6xdB-w-tVsSC-15W27bBg)**

### Pipeline Stages

```
 ╔═══════════════════════════════════════════════════════════════╗
 ║                     DATA SOURCES                             ║
 ║  ┌─────────┐  ┌──────────┐  ┌─────────────┐  ┌───────────┐ ║
 ║  │  CSV /   │  │Timescale │  │  Open-Meteo  │  │  Excel    │ ║
 ║  │  Files   │  │   DB     │  │  Weather API │  │  .xlsx    │ ║
 ║  └────┬─────┘  └────┬─────┘  └──────┬──────┘  └─────┬─────┘ ║
 ╚═══════╪═════════════╪═══════════════╪════════════════╪═══════╝
         └──────────┬──┴───────────────┴────────────────┘
                    ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  STAGE 1: DataCleaner (Fluent Builder)                       │
 │                                                              │
 │  .load()  →  .sanitize(strategy)  →  .set_datetime_index()  │
 │           →  .to_epoch_ms()       →  .result()               │
 │                                                              │
 │  Strategies: "interpolate" (linear + fill)                   │
 │              "ffill" (forward-fill + backfill leading NaNs)  │
 └──────────────────────────┬───────────────────────────────────┘
                            ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  STAGE 2: Train / Test Split                                 │
 │                                                              │
 │  train = series[:-horizon]                                   │
 │  test  = series[-horizon:]   (holdout for evaluation)        │
 └──────────────────────────┬───────────────────────────────────┘
                            ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  STAGE 3: Optional Transforms & Feature Engineering          │
 │                                                              │
 │  Variance Stabilisation:  Log | Sqrt | Box-Cox (λ via MLE)  │
 │                                                              │
 │  Feature Matrix (for ML models):                             │
 │  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐   │
 │  │ Lag features   │  │ Rolling stats  │  │ Fourier      │   │
 │  │ lag_1..lag_12  │  │ mean/std by    │  │ sin/cos      │   │
 │  │ via shift(k)   │  │ window [3,6,12]│  │ K harmonics  │   │
 │  └────────────────┘  └────────────────┘  └──────────────┘   │
 │           └──────────────┴────────────────────┘              │
 │                          ▼                                   │
 │               build_supervised_matrix()                      │
 └──────────────────────────┬───────────────────────────────────┘
                            ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  STAGE 4: Forecasting Engine                                 │
 │                                                              │
 │  model.fit_predict(train, horizon=H)                         │
 │                                                              │
 │  ┌─────────────┐ ┌────────────┐ ┌──────────┐ ┌───────────┐ │
 │  │ Classical   │ │ ML         │ │ Neural   │ │ Foundation│ │
 │  │ ARIMA, ETS  │ │ XGBoost    │ │ NHITS    │ │ TimeGPT   │ │
 │  │ Holt-Winters│ │ Prophet    │ │ MLP      │ │ (zero-    │ │
 │  │             │ │ VAR        │ │          │ │  shot)    │ │
 │  └─────────────┘ └────────────┘ └──────────┘ └───────────┘ │
 │                          │                                   │
 │                          ▼                                   │
 │               ┌──────────────────┐                           │
 │               │  ForecastResult  │   ← Uniform output       │
 │               │  .forecast       │      from ALL models      │
 │               │  .lower / .upper │                           │
 │               │  .duration_ms    │                           │
 │               │  .info_criteria  │                           │
 │               │  .correlation_id │                           │
 │               └──────────────────┘                           │
 └──────────────────────────┬───────────────────────────────────┘
                            ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  STAGE 5: Evaluation & Comparison                            │
 │                                                              │
 │  ModelComparison(test)                                       │
 │    .add(result_arima)                                        │
 │    .add(result_xgboost)                                      │
 │    .add(result_prophet)                                      │
 │    .leaderboard(sort_by="RMSE")                              │
 │    .metric_deltas(baseline="AutoARIMA")                      │
 │                                                              │
 │  Computes: MAE | RMSE | MAPE | sMAPE | R²                   │
 └──────────────────────────┬───────────────────────────────────┘
                            ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  STAGE 6: Output Artifacts                                   │
 │                                                              │
 │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
 │  │ Leaderboard  │  │  Plots       │  │  AnalyticsEngine  │  │
 │  │ .json + HTML │  │  Forecast    │  │  Client            │  │
 │  │              │  │  Seasonal    │  │  .log_run()        │  │
 │  │              │  │  ACF/PACF    │  │  .to_json()        │  │
 │  │              │  │  Decomp      │  │  UUID traceability │  │
 │  └──────────────┘  └──────────────┘  └───────────────────┘  │
 └──────────────────────────────────────────────────────────────┘
```

### Traceability Thread

Every stage is tagged with a **UUID v4 correlation ID** (`CONFIG.correlation_id`) that propagates through:

```
DataCleaner → Forecaster → ForecastResult → ModelComparison → AnalyticsEngineClient → JSON Export
```

This enables end-to-end audit trail across distributed system logs.

---

## 4. Forecasting Algorithm Comparison

### Interactive Diagram

**[View Model Hierarchy on Excalidraw](https://excalidraw.com/#json=tDY3C33kFr8SUdz-bOU9Y,4kzK6aF-nYfq8lFbp1GMlA)**

### Algorithm Matrix

| Model | Paradigm | Backend | Training Required | Prediction Intervals | Best For | Limitation |
|-------|----------|---------|-------------------|---------------------|----------|------------|
| **AutoARIMA** | Classical | pmdarima | Yes (seconds) | Native (analytical) | Strong seasonal patterns with known period | Assumes linearity & stationarity |
| **AutoETS** | Classical | statsmodels | Yes (seconds) | Native (state space) | Smooth trends with seasonal decomposition | No exogenous regressors |
| **Holt-Winters** | Classical | statsmodels | Yes (seconds) | Simulation-based | Stable additive/multiplicative seasonality | Fixed seasonal pattern |
| **VAR** | Multivariate | statsmodels | Yes (seconds) | Impulse-response | Cross-variable lead/lag relationships | Requires stationarity across all series |
| **XGBoost** | Machine Learning | xgboost | Yes (seconds–minutes) | Quantile regression (3 models) | Non-linear patterns, feature-rich data | Recursive prediction can compound errors |
| **Prophet** | Machine Learning | prophet | Yes (minutes) | Posterior sampling | Irregular seasonality, trend breaks, holidays | Slower training, daily/sub-daily focus |
| **NHITS** | Deep Learning | neuralforecast | Yes (minutes–hours) | Not provided | Long-horizon, complex multi-scale patterns | Needs significant training data |
| **MLP** | Deep Learning | neuralforecast | Yes (minutes–hours) | Not provided | General-purpose neural baseline | Needs significant training data |
| **TimeGPT** | Foundation Model | nixtla API | **No (zero-shot)** | API-provided | Rapid prototyping, cold-start scenarios | External API dependency, cost per call |

### How Each Model Generates Forecasts

#### Classical Models (AutoARIMA, AutoETS, Holt-Winters)

```
Training Series  →  Fit parametric model (ARIMA orders / ETS components)
                 →  Analytical or simulation-based h-step forecast
                 →  Prediction intervals from model variance estimates
```

- **AutoARIMA:** Automatically selects (p,d,q)(P,D,Q)m orders via stepwise AICc minimisation. Uses `pmdarima.auto_arima()`.
- **AutoETS:** Grid search over 18 Error/Trend/Seasonal combinations (3×3×2), selects by AIC.
- **Holt-Winters:** Exponential smoothing with explicit additive or multiplicative seasonal component.
- **auto_select_univariate():** Runs all three, returns the model with lowest AICc.

#### Machine Learning Models (XGBoost, Prophet)

```
Training Series  →  build_supervised_matrix()  →  Feature matrix (lags + rolling + Fourier)
                 →  Fit gradient-boosted trees (3 regressors)
                 →  Recursive multi-step prediction (each step feeds back)
```

**XGBoost internals:**
1. Constructs features: `lag_1..lag_12` + `roll_mean_3/6/12` + `roll_std_3/6/12` + `sin/cos` Fourier harmonics
2. Fits **3 XGBRegressor models** simultaneously:
   - Point estimate: `reg:squarederror`
   - Lower bound: `reg:quantileerror` (α/2 = 0.025)
   - Upper bound: `reg:quantileerror` (1-α/2 = 0.975)
3. Predicts recursively: each forecast step is appended to history so lag features remain valid

**Prophet internals:**
- Decomposes series into: `y(t) = g(t) + s(t) + h(t) + ε(t)` (trend + seasonality + holidays + error)
- Automatic changepoint detection for trend breaks
- Fourier-based seasonality modelling
- Posterior sampling for uncertainty quantification

#### Multivariate (VAR)

```
Multi-column DataFrame  →  Granger causality tests (identify lead/lag)
                        →  Fit VAR(p) with IC-based lag selection
                        →  h-step forecast with impulse-response intervals
```

- Runs `grangercausalitytests()` on all column pairs
- Results stored in `result.extra["granger"]` as `{cause→effect: min_p_value}`

#### Neural & Foundation Models

```
Training Series  →  Convert to long format (unique_id, ds, y)
                 →  NHITS/MLP: Train for max_steps epochs
                 →  TimeGPT: Send to Nixtla API (zero-shot)
                 →  Return forecast series
```

### Decision Framework: When to Use What

```
                    ┌─────────────────────────┐
                    │  Do you have training    │
                    │  data & time to train?   │
                    └───────┬─────────┬────────┘
                       No   │         │  Yes
                            ▼         ▼
                    ┌──────────┐   ┌──────────────────────┐
                    │ TimeGPT  │   │ Is the relationship  │
                    │(zero-    │   │ linear & seasonal?   │
                    │ shot)    │   └───────┬────────┬─────┘
                    └──────────┘      Yes  │        │ No
                                          ▼        ▼
                               ┌───────────┐  ┌─────────────────┐
                               │ Classical  │  │ Multiple related │
                               │ AutoARIMA  │  │ series?          │
                               │ AutoETS    │  └────┬──────┬─────┘
                               │ Holt-      │  Yes  │      │ No
                               │ Winters    │       ▼      ▼
                               └───────────┘  ┌──────┐  ┌──────────┐
                                              │ VAR  │  │ XGBoost  │
                                              │      │  │ Prophet  │
                                              └──────┘  │ NHITS    │
                                                        └──────────┘
```

---

## 5. KPI Metrics — Deep Dive

### Interactive Diagram

**[View Evaluation Pipeline on Excalidraw](https://excalidraw.com/#json=IbDEGJdWuz1y2huDydQpH,mN27lFRAjLwPyxtUoNNZ_Q)**

### The 5 Evaluation Metrics

#### MAE — Mean Absolute Error

```
MAE = (1/n) × Σ |yᵢ − ŷᵢ|
```

- **What it measures:** Average forecast error in the **original units** of the data
- **Interpretation:** "On average, our forecast is off by X units"
- **Strengths:** Intuitive, robust to outliers (compared to RMSE), directly actionable
- **Use when:** You need a simple, interpretable error measure; all errors are equally costly

---

#### RMSE — Root Mean Square Error

```
RMSE = √( (1/n) × Σ (yᵢ − ŷᵢ)² )
```

- **What it measures:** Average error magnitude, **penalising large deviations** disproportionately
- **Interpretation:** "Our typical error is X units, but large misses cost more"
- **Strengths:** Differentiable (good for optimisation), penalises spikes
- **Use when:** Large forecast errors have outsized business impact (e.g. energy grid balancing, inventory)
- **Relationship to MAE:** RMSE ≥ MAE always. The gap indicates error variance — if RMSE >> MAE, you have occasional large misses

---

#### MAPE — Mean Absolute Percentage Error

```
MAPE = (100/n) × Σ |yᵢ − ŷᵢ| / |yᵢ|
```

- **What it measures:** Average **percentage** forecast error
- **Interpretation:** "Our forecast is off by X% on average"
- **Strengths:** Scale-independent — enables comparison across different series with different units
- **Use when:** Comparing forecast quality across products, regions, or time granularities
- **Warning:** Undefined/unreliable when actual values approach zero (division by near-zero). The toolkit warns when |y| < 1e-8

---

#### sMAPE — Symmetric Mean Absolute Percentage Error

```
sMAPE = (100/n) × Σ |yᵢ − ŷᵢ| / ((|yᵢ| + |ŷᵢ|) / 2)
```

- **What it measures:** Percentage error that treats **over-forecasts and under-forecasts symmetrically**
- **Interpretation:** "Our percentage error is X%, balanced between over and under"
- **Strengths:** Bounded (0–200%), handles near-zero actuals better than MAPE
- **Use when:** Data contains periods of very low values; you want a fairer percentage metric
- **Advantage over MAPE:** MAPE penalises over-forecasts more heavily than under-forecasts; sMAPE corrects this asymmetry

---

#### R² — Coefficient of Determination

```
R² = 1 − (SS_res / SS_tot)

where SS_res = Σ (yᵢ − ŷᵢ)²     (residual sum of squares)
      SS_tot = Σ (yᵢ − ȳ)²       (total sum of squares)
```

- **What it measures:** **Proportion of variance** in the data explained by the model
- **Interpretation:** R²=0.85 means "the model explains 85% of the variability in the actual values"
- **Scale:** 1.0 = perfect; 0.0 = no better than predicting the mean; negative = worse than the mean
- **Use when:** You need a single number summarising overall model quality
- **Caveat:** Can be misleading for non-stationary series or when comparing across different test sets

### Metric Selection Guide

| Business Question | Primary Metric | Secondary |
|-------------------|---------------|-----------|
| "How far off are we in real units?" | **MAE** | RMSE |
| "How costly are our worst misses?" | **RMSE** | MAE |
| "What's our % accuracy across products?" | **MAPE** | sMAPE |
| "% accuracy with near-zero values?" | **sMAPE** | MAE |
| "Overall model quality score?" | **R²** | RMSE |
| "Which model is best overall?" | **RMSE** (default sort) | All 5 |

### How ModelComparison Works

```python
cmp = ModelComparison(test_series)

cmp.add(result_arima)       # ForecastResult from AutoARIMA
cmp.add(result_xgboost)     # ForecastResult from XGBoost
cmp.add(result_prophet)     # ForecastResult from Prophet

# Ranked table with all 5 KPIs + runtime + info criteria
cmp.leaderboard(sort_by="RMSE")

# Relative improvement vs baseline
cmp.metric_deltas(baseline_model="AutoARIMA")

# Overlay all forecasts on one plot
cmp.plot_all(observed_full, train_end=split_date)
```

Output: styled HTML leaderboard in Jupyter, JSON export for pipelines.

---

## 6. Current Toolkit Status

### Module Readiness

| Module | Files | Size | Status | Completeness |
|--------|-------|------|--------|-------------|
| **config** | 1 | 2.3 KB | Production | Singleton, env-var driven, UUID traceability |
| **connectivity** | 4 | 21.6 KB | Production | CSV, Excel, TimescaleDB, Open-Meteo API |
| **cleaning** | 2 | 4.5 KB | Production | Fluent API, interpolate/ffill strategies |
| **signals** | 1 | 5.6 KB | Production | Log, Sqrt, Box-Cox with auto-offset |
| **features** | 3 | 11.6 KB | Production | Lags, rolling stats, Fourier, supervised matrix |
| **forecasting** | 5 | 41.4 KB | Production | 9 models across 4 paradigms |
| **evaluation** | 3 | 10.1 KB | Production | 5 KPIs, leaderboard, traceability client |
| **viz** | 5 | 12.1 KB | Production | 5 plot types, styled HTML, metric cards |
| **display** | 1 | 0.2 KB | Production | Re-export layer |
| **experiment runner** | 1 | — | Production | YAML-driven, checkpoint/resume capable |

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total Python modules | 35 |
| Estimated LOC | ~3,000 |
| Forecasting models | 9 |
| Evaluation KPIs | 5 |
| Data source types | 4 (CSV, Excel, TimescaleDB, Open-Meteo) |
| Variance transforms | 3 (Log, Sqrt, Box-Cox) |
| Feature components | 3 (Lags, Rolling, Fourier) |
| Visualization types | 5 |
| Design patterns | 6 (Fluent, Strategy, Factory, Singleton, Visitor, CorrelationID) |
| Python version | ≥ 3.10 |
| Test coverage | **0% — No test suite** |
| CI/CD pipeline | **None** |

### Risk & Gap Assessment

| Area | Status | Risk | Recommendation |
|------|--------|------|----------------|
| Core functionality | All modules production-ready | Low | — |
| Test coverage | **No tests** | **High** | Add pytest suite with unit + integration tests |
| CI/CD | **No pipeline** | **High** | Add GitHub Actions for lint, test, build |
| Documentation | Comprehensive (README, ONBOARDING, ARCHITECTURE) | Low | — |
| Dependency management | setup.py with optional extras | Low | Consider pyproject.toml migration |
| Deployment | K8s YAML + Docker Compose placeholder | Medium | Flesh out Docker build, add Helm chart |

---

## 7. Design Patterns

### Interactive Diagram

**[View Design Patterns on Excalidraw](https://excalidraw.com/#json=NXa2eVvbfBptGZGBFDNb9,zXs85MnAvQe5D28fAinTtQ)**

| Pattern | Implementation | Business Value |
|---------|---------------|---------------|
| **Fluent Builder** | `DataCleaner.load().sanitize().result()` | Readable, chainable data pipelines |
| **Strategy** | `LogTransform` / `SqrtTransform` / `BoxCoxTransform` | Pluggable transforms without code changes |
| **Abstract Factory** | `BaseForecaster` → 9 concrete models | Drop-in model swapping, uniform API |
| **Singleton** | `CONFIG` | Consistent global state, single source of truth |
| **Visitor/Aggregator** | `ModelComparison.add().leaderboard()` | Heterogeneous model evaluation |
| **Correlation ID** | UUID v4 across all modules | End-to-end audit trail |

---

## 8. Experiment Orchestration

### Interactive Diagram

**[View Experiment Runner on Excalidraw](https://excalidraw.com/#json=15v3fu0fGzV0OJ_AB2p7N,RZ74uM9kbSdaVENfq2swLw)**

### Workflow

```bash
# Fresh experiment
python scripts/toolkit_demo.py --params configs/params.yaml --verbose

# Resume interrupted run
python scripts/toolkit_demo.py --params configs/params.yaml --resume experiments/run_dir/
```

### Output Structure

```
experiments/{name}_{timestamp}_{uuid}/
├── params.yaml              # Frozen config snapshot
├── checkpoints/             # Top-N model pickles by metric
│   ├── best_checkpoints.json
│   └── checkpoint_arima_312.45.pkl
├── runs/                    # Per-model run records (.json)
├── test_runs/               # Test-set evaluation results
├── logs/                    # Structured per-model log files
├── plots/                   # Forecast visualisation PNGs
└── results/
    ├── leaderboard.json     # Ranked model comparison
    └── summary.json         # Experiment metadata
```

**Key Features:**
- Checkpoint-based resumption (skip completed models)
- Top-N best model retention (pruned by metric)
- Sprint snapshots at configurable intervals
- Structured logging per model with timestamps

---

## 9. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Core** | pandas, numpy, scipy | Data manipulation, numerical computation |
| **Statistical** | statsmodels, pmdarima | ARIMA, ETS, Holt-Winters, VAR |
| **ML** | xgboost, prophet | Gradient boosting, trend decomposition |
| **Neural** | neuralforecast (PyTorch) | NHITS, MLP deep learning models |
| **Foundation** | nixtla | TimeGPT zero-shot forecasting |
| **Database** | SQLAlchemy, psycopg2 | TimescaleDB connectivity |
| **Weather** | Open-Meteo API | Free weather data (no API key) |
| **Visualization** | matplotlib, IPython | Plots, styled HTML tables |
| **Deployment** | Kubernetes, Docker | GPU cluster job execution |

---

## 10. Next Steps & Roadmap Recommendations

### Immediate (Critical)

1. **Add test suite** — pytest with unit tests for metrics, transforms, feature engineering, and integration tests for the forecasting pipeline
2. **Set up CI/CD** — GitHub Actions for linting (ruff/flake8), testing, and package build on every PR

### Short-term (High Value)

3. **Migrate to pyproject.toml** — Modern Python packaging standard
4. **Dockerize the toolkit** — Production-ready container image with all dependencies
5. **Add backtesting** — Walk-forward cross-validation for more robust model evaluation

### Medium-term (Strategic)

6. **Model registry integration** — MLflow or similar for experiment tracking beyond local JSON
7. **API service layer** — FastAPI wrapper for serving forecasts as REST endpoints
8. **Streaming support** — Real-time forecast updates for live data feeds
