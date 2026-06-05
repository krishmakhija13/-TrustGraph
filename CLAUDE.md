# TrustGraph — Credit Risk Scoring Project

## Project Overview
End-to-end credit default prediction pipeline using the Home Credit dataset.
Goal: predict whether a loan applicant will default (TARGET=1).

## Folder Structure
```
TrustGraph/
├── CLAUDE.md
├── requirements.txt
├── data/
│   ├── raw/                  # Raw input CSVs (application_train.csv, etc.)
│   └── processed/
│       ├── baseline_results.csv
│       └── eda_charts/       # Saved plots (.png)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   └── 02_baseline.ipynb     # Baseline model training & evaluation
├── models/                   # Serialized model artifacts
└── src/                      # Reusable Python modules
```

## Data
- Primary file: `data/raw/application_train.csv`
- Source: Home Credit Default Risk (Kaggle)
- Target column: `TARGET` (1 = default, 0 = no default)

## Notebooks
| Notebook | Purpose |
|---|---|
| 01_eda.ipynb | Data profiling, missing-value analysis, distribution plots, correlation heatmap |
| 02_baseline.ipynb | LightGBM + Logistic Regression baseline, AUC-ROC, KS statistic, ROC curve plot |

## Key Conventions
- All file paths are relative to project root
- Random seed: 42
- Test size: 20% stratified split
- Metrics: AUC-ROC (primary), KS statistic (secondary)
