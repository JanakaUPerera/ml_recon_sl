# Sri Lanka Household Electricity Demand Forecasting

> Advanced time series forecasting pipeline built on raw multi-city Sri Lankan smart-meter data. Converts ~5.4 GB of per-household cumulative counter reads into a national daily demand series, then compares SARIMA, LSTM/GRU, and XGBoost forecasts, plus GARCH(1,1) volatility modelling of demand shocks around the 2023 tariff-revision period.

**Academic context:** NIB 7072 Machine Learning — MSc Data Science, Coventry University / NIBM Sri Lanka (2025)
**Portfolio context:** Designed as a Data Engineer portfolio project demonstrating time series forecasting and data engineering on raw sensor data.

---

## Problem Statement

This project forecasts *how much electricity a population will consume*. The dataset combines 20 months of smart-meter and billing consumption records with three waves of household surveys across Sri Lankan cities. The raw smart-meter files store **cumulative counter readings** (like an odometer), not ready-to-model interval consumption — the core engineering task is building a clean, differenced, aggregated daily demand series before any forecasting can happen.

| Property | Value |
|---|---|
| Dataset | Residential Electricity Consumption: Multi-round Longitudinal Surveys + Energy Provider Data (Sri Lanka) |
| Raw dataset size | ~5.4 GB across 9 CSV files |
| Households | ~4,063 (non-smart meter) / ~4,000 survey panel |
| Smart-meter coverage | 6-hour interval: 2023-01-01 → 2024-12-31 (continuous, 5 files). 15-min interval: 2023-10-01 → 2024-12-23 (continuous, 3 files) |
| Forecasting target | National daily aggregate residential electricity consumption (kWh), derived by differencing cumulative smart-meter reads and summing across households |
| Prediction horizon | 30 days ahead |
| Volatility target | Daily log-returns of the aggregate series (and peak-tariff `TR2` sub-series), analysed for conditional heteroscedasticity around 2023 tariff-hike shocks |

---

## Why this is a strong Data Engineer portfolio piece

Most public time-series coursework projects use an already-clean CSV (one row per day, one target column). This dataset forces real ETL work first:

1. **Chunked ingestion** of multi-GB files (individual files up to 1.9 GB) — no naive full-memory load.
2. **Cumulative-to-interval conversion**: per-household sort + diff of `TOTAL_IMPORT (kWh)`, since raw values only ever increase.
3. **Meter-reset handling**: a negative diff is a counter reset artefact, not negative consumption — must be detected and excluded, not silently summed.
4. **Deduplication** of repeated/out-of-order timestamp reads.
5. **Aggregation** across thousands of households into a single, clean, forecastable daily series.

The forecasting and GARCH modelling that follows sits on top of a pipeline that looks like production smart-meter/grid data engineering, not a spreadsheet exercise.

---

## Project Structure

```
ml_recon_sl/
│
├── notebooks/
│   ├── 01_dataset_understanding.ipynb        # Schema, multi-file continuity, target/horizon decision
│   ├── 02_preprocessing_feature_engineering.ipynb  # ETL, differencing, decomposition, lag features
│   ├── 03_model_development.ipynb            # SARIMA, LSTM/GRU, XGBoost, walk-forward CV
│   ├── 04_garch_volatility.ipynb             # GARCH(1,1) volatility clustering analysis
│   └── 05_comparison_reflection.ipynb        # Model comparison, critical reflection
│
├── src/
│   ├── pipeline/
│   │   ├── data_loader.py                    # Chunked CSV ingestion
│   │   ├── etl_smart_meter.py                # Cumulative-counter differencing + aggregation
│   │   └── feature_engineering.py            # Lags, rolling stats, calendar features
│   └── models/
│       ├── sarima_model.py
│       ├── lstm_model.py
│       └── xgboost_model.py
│
├── tests/
│   └── test_pipeline.py
│
├── outputs/
│   ├── figures/                              # gitignored — regenerated from notebooks
│   ├── models/                               # gitignored
│   └── reports/                              # gitignored
│
├── dataset/                                  # gitignored — ~5.4 GB raw data (not committed)
├── docs/                                     # gitignored — literature review notes (internal)
├── requirements.txt
├── WORK_PLAN.md                              # gitignored — internal phase-by-phase plan
└── STUDY_GUIDE.md                            # committed — results & reasoning, written per phase
```

---

## Pipeline Status

| Task | Key Decisions |
|---|---|
| 1. Dataset Understanding & Literature Review | Target/horizon defined, multi-file continuity confirmed |
| 2. Time Series Exploration & Preprocessing | ETL for cumulative-counter differencing, decomposition, lag/rolling features |
| 3. Model Development | SARIMA, LSTM/GRU, XGBoost with walk-forward CV |
| 4. GARCH Volatility Modelling | GARCH(1,1) on daily demand returns |
| 5. Comparison & Critical Reflection | Cross-model comparison, business/ethical discussion |

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Data processing | Python, Pandas, NumPy, PyArrow (chunked ingestion for multi-GB CSVs) |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Statistical forecasting | statsmodels, pmdarima (SARIMA / auto_arima) |
| Deep learning | TensorFlow / Keras (LSTM / GRU) |
| ML-based forecasting | XGBoost, scikit-learn |
| Volatility modelling | `arch` (GARCH) |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow |

---

## Setup

```bash
# From the repo root
cd ml_recon_sl

# Create and activate virtual environment
python -m venv ml-venv
ml-venv\Scripts\activate          # Windows
# source ml-venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Dataset already present under dataset/ (gitignored — not committed)

# Run notebooks in order
jupyter notebook notebooks/
```

---

*MSc Data Science — Coventry University / NIBM Sri Lanka, 2025*
