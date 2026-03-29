# TSAtoolkit Architecture

Interactive architecture diagrams for the `aic_ts_suite` codebase. Each diagram is hosted on Excalidraw and can be viewed, edited, and shared.

## Diagrams

### 1. Module Architecture Overview

High-level view of all package modules organized by layer: Data Ingestion, Transform & Features, Forecasting Engine, and Evaluation & Output.

[View on Excalidraw](https://excalidraw.com/#json=JRc6HIQ6JCYY6Hgb25AjN,SeNIBmzVCsEGjPgApUiVAg)

**Layers:**
- **Data Ingestion** (blue) — `config`, `connectivity`, `cleaning`
- **Transform & Features** (purple) — `signals`, `features`
- **Forecasting Engine** (green) — `univariate`, `multivariate`, `ml_models`, `neural`
- **Evaluation & Output** (pink/orange) — `evaluation`, `viz`, `display`

---

### 2. Data Pipeline Flow

End-to-end data flow from ingestion sources (CSV, TimescaleDB, Open-Meteo) through cleaning, transformation, forecasting, and evaluation to final outputs.

[View on Excalidraw](https://excalidraw.com/#json=I18_GDCG_l5pUvUtMCJi0,E6xdB-w-tVsSC-15W27bBg)

**Flow:**
```
Data Sources -> DataCleaner -> Train/Test Split -> Transforms -> Features -> Forecaster -> ModelComparison -> Outputs
```

---

### 3. Forecasting Model Hierarchy

Class hierarchy showing `BaseForecaster` ABC and all 9 concrete implementations across 4 paradigms.

[View on Excalidraw](https://excalidraw.com/#json=tDY3C33kFr8SUdz-bOU9Y,4kzK6aF-nYfq8lFbp1GMlA)

**Model categories:**
| Category | Models | Backend |
|----------|--------|---------|
| Classical/Univariate | AutoARIMA, AutoETS, Holt-Winters | pmdarima, statsmodels |
| Multivariate | VAR + Granger causality | statsmodels |
| Machine Learning | XGBoost, Prophet | xgboost, prophet |
| Neural/Foundation | N-HiTS, MLP, TimeGPT | neuralforecast, nixtla |

---

### 4. Evaluation & Visualization Pipeline

How `ForecastResult` objects flow into `ModelComparison`, are scored with 5 KPIs, rendered as plots, and traced via `AnalyticsEngineClient`.

[View on Excalidraw](https://excalidraw.com/#json=IbDEGJdWuz1y2huDydQpH,mN27lFRAjLwPyxtUoNNZ_Q)

**Metrics:** MAE, RMSE, MAPE, sMAPE, R-squared

**Viz:** `plot_forecast`, `plot_seasonal`, `plot_acf_pacf`, `plot_decomposition`, `styled_summary`

---

### 5. Experiment Runner (CLI)

Architecture of `scripts/toolkit_demo.py` — YAML-driven experiment orchestration with checkpointing, logging, and artifact management.

[View on Excalidraw](https://excalidraw.com/#json=15v3fu0fGzV0OJ_AB2p7N,RZ74uM9kbSdaVENfq2swLw)

**Experiment folder structure:**
```
experiments/{name}_{timestamp}_{id}/
  checkpoints/    # Best-N model .pkl files
  runs/           # Per-model run .json artifacts
  test_runs/      # Test evaluation results
  logs/           # Structured log files
  results/        # leaderboard.json, summary.json
  params.yaml     # Config snapshot
```

---

### 6. Design Patterns

Key software patterns used throughout the codebase: Fluent Builder, Strategy, Abstract Factory, Singleton, Visitor/Aggregator, and Correlation ID traceability.

[View on Excalidraw](https://excalidraw.com/#json=NXa2eVvbfBptGZGBFDNb9,zXs85MnAvQe5D28fAinTtQ)

| Pattern | Where | Purpose |
|---------|-------|---------|
| Fluent Builder | `DataCleaner` | Chainable preprocessing pipeline |
| Strategy | `BaseTransform` | Pluggable variance-stabilizing transforms |
| Abstract Factory | `BaseForecaster` | Uniform interface for all model types |
| Singleton | `CONFIG` | Global settings + correlation ID |
| Visitor/Aggregator | `ModelComparison` | Collect and rank heterogeneous results |
| Correlation ID | All modules | End-to-end run traceability (UUID v4) |
