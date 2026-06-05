# TrustGraph — Explainable Credit Risk Scoring for Thin-File Borrowers

> An end-to-end credit default prediction pipeline with dual-model architecture,
> SHAP explainability, fairness auditing, and a live Streamlit demo — built for
> the 400 million credit-invisible individuals in India and emerging markets.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-f97316?style=flat-square)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-8b5cf6?style=flat-square)](https://shap.readthedocs.io/)
[![Fairlearn](https://img.shields.io/badge/Fairness-Fairlearn-0ea5e9?style=flat-square)](https://fairlearn.org/)
[![SSRN](https://img.shields.io/badge/Paper-SSRN%20(forthcoming)-64748b?style=flat-square)](https://ssrn.com)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## The Problem

India has **400 million credit-invisible individuals** — people who exist outside the formal
credit bureau system and are systematically excluded from affordable loan products despite
often having stable incomes and genuine creditworthiness.
Of India's **63 million MSMEs**, fewer than 15% have ever accessed formal credit, not because
they are high-risk, but because traditional scorecards cannot evaluate them — there is simply
no bureau history to score.

TrustGraph addresses this by building an alternative credit scoring engine that combines
machine-learning predictions with interpretable scorecards, behavioural cash-flow signals,
and explicit fairness constraints — producing scores that lenders can trust, regulators can
audit, and borrowers can understand.

---

## What TrustGraph Builds

- **Dual-Model Architecture** — a production-grade LightGBM (AUC 0.7707, Gini 0.5414)
  running alongside a fully interpretable WOE scorecard (AUC 0.7286), giving lenders
  a choice between raw accuracy and regulatory auditability on every decision.

- **India Trust Layer** — a synthetic cash-flow generator (`src/india_layer.py`) that
  simulates three thin-file borrower archetypes (kirana shop owner, gig worker, small
  farmer) with realistic Indian economic parameters, maps them to the TrustGraph feature
  schema, and scores them through both models in real time.

- **Explainability + Fairness** — SHAP beeswarm, waterfall, and bar plots for global and
  individual explanations; Fairlearn-based gender-fairness audit (demographic parity
  difference reduced from 0.1535 → 0.0025 via ThresholdOptimizer); plain-English
  reason-code generator for loan officers.

- **Live Streamlit Demo** — an interactive scoring UI (`app/streamlit_app.py`) with
  borrower profile sliders, one-click India presets, side-by-side model comparison,
  colour-coded decisions, and live fairness disclosure — launchable with a single command.

---

## Results

### Model Performance (Home Credit Open Dataset — 20% held-out validation set)

| Notebook | Model | AUC-ROC | Gini | KS Statistic | Notes |
|---|---|---|---|---|---|
| 02 | Logistic Regression (baseline) | 0.6184 | 0.2368 | 0.1750 | Raw features, no engineering |
| 02 | LightGBM (baseline) | 0.7611 | 0.5222 | 0.3908 | Raw features, no engineering |
| 03 | LightGBM + Feature Engineering | 0.7689 | 0.5378 | 0.4032 | +8 domain features |
| **04** | **LightGBM Production** | **0.7707** | **0.5414** | **0.4049** | Tuned params, early stop @ tree 514 |
| 05 | WOE Scorecard | 0.7286 | 0.4571 | 0.3434 | 10 features, fully interpretable |

### Feature Engineering Lift (Notebook 03 vs 02)

| Metric | Baseline | + Engineering | Delta |
|---|---|---|---|
| AUC-ROC | 0.7611 | 0.7689 | **+0.0078** |
| KS Statistic | 0.3908 | 0.4032 | **+0.0124** |

### Fairness Audit — Gender (Notebook 08, Fairlearn)

| Metric | Pre-Mitigation | Post-Mitigation |
|---|---|---|
| Demographic Parity Difference | 0.1535 | **0.0025** |
| Equalized Odds Difference | 0.1674 | 0.0187 |
| Selection Rate (Male) | 0.3784 | 0.2575 |
| Selection Rate (Female) | 0.2249 | 0.2551 |

Mitigation method: Fairlearn `ThresholdOptimizer` (demographic parity constraint, post-processing).

### Model Stability — PSI (Notebook 09, Gaussian noise std=0.1)

| Feature | PSI | Status |
|---|---|---|
| EXT_SOURCE_3 | 0.5294 | ⚠️ UNSTABLE — monitor monthly |
| CREDIT_TERM | 0.0589 | ✅ Stable |
| All others | < 0.006 | ✅ Stable |

AUC under simulated distribution shift: **0.7608** (−0.0099 from baseline) — robust overall.

---

## Architecture

```
Raw Data (Home Credit CSV)
        │
        ▼
┌───────────────────┐
│  01_eda.ipynb     │  Data profiling, missing-value analysis,
│                   │  distribution plots, correlation heatmap
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  03_features.ipynb│  8 domain-engineered features:
│                   │  CREDIT_INCOME_RATIO, EXT_SOURCE_MEAN,
│                   │  DAYS_EMPLOYED_RATIO, FLAG_MISSING_EXT …
└─────────┬─────────┘
          │
          ├──────────────────────┐
          ▼                      ▼
┌─────────────────┐    ┌─────────────────────┐
│  04_gbm.ipynb   │    │  05_scorecard.ipynb  │
│  LightGBM       │    │  WOE Scorecard       │
│  AUC 0.7707     │    │  AUC 0.7286          │
│  Gini 0.5414    │    │  Fully auditable     │
└────────┬────────┘    └──────────┬──────────┘
         │                        │
         └──────────┬─────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Trust Layer          │
        │  06_india_layer.ipynb │  Synthetic thin-file profiles
        │  07_explainability    │  SHAP + reason codes
        │  08_fairness          │  Fairlearn bias audit
        │  09_stability         │  PSI monitoring
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Streamlit Demo       │
        │  app/streamlit_app.py │  Interactive scoring UI
        └───────────────────────┘
```

---

## Project Structure

```
TrustGraph/
├── CLAUDE.md                          # Project conventions for Claude Code
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── app/
│   └── streamlit_app.py               # Interactive Streamlit demo
│
├── data/
│   ├── raw/                           # ⚠ Not tracked in git — add CSV here
│   │   └── application_train.csv      # Home Credit dataset (Kaggle)
│   └── processed/                     # Generated outputs — tracked in git
│       ├── features_train.csv         # Engineered feature matrix (307k rows)
│       ├── baseline_results.csv       # Notebook 02 metrics
│       ├── feature_results.csv        # Notebook 03 metrics
│       ├── gbm_results.csv            # Notebook 04 metrics
│       ├── scorecard_table.csv        # WOE bins + points (Notebook 05)
│       ├── india_profiles.csv         # Synthetic India borrowers (06)
│       ├── india_decisions.csv        # Scoring decisions (06)
│       ├── reason_codes.csv           # SHAP reason codes (07)
│       ├── fairness_results.csv       # Gender fairness metrics (08)
│       └── stability_results.csv      # PSI results (09)
│       └── eda_charts/                # All saved plots (.png)
│
├── models/                            # ⚠ Not tracked in git — regenerate
│   ├── lgbm_final.pkl                 # Production LightGBM (3.64 MB)
│   └── scorecard_final.pkl            # WOE Scorecard + LR model
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_baseline.ipynb              # LightGBM + LogReg baselines
│   ├── 03_features.ipynb              # Feature engineering (8 features)
│   ├── 04_gbm.ipynb                   # Production LightGBM
│   ├── 05_scorecard.ipynb             # WOE scorecard
│   ├── 06_india_layer.ipynb           # Synthetic India profiles [SYNTHETIC]
│   ├── 07_explainability.ipynb        # SHAP analysis + reason codes
│   ├── 08_fairness.ipynb              # Fairlearn gender audit
│   └── 09_stability.ipynb             # PSI model stability
│
└── src/
    ├── india_layer.py                 # Synthetic cash-flow generator [SYNTHETIC]
    ├── explainability.py              # Reason-code generator
    └── print_summary.py               # Cross-notebook summary script
```

---

## How to Run

### Prerequisites
- Python 3.10+
- [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data) from Kaggle (`application_train.csv`, ~166 MB)

### Steps

**1. Clone and install**
```bash
git clone https://github.com/<your-username>/TrustGraph.git
cd TrustGraph
pip install -r requirements.txt
```

**2. Add the data**
```bash
# Place the downloaded CSV here:
data/raw/application_train.csv
```

**3. Run notebooks in order**
```bash
# Open Jupyter and run each notebook top-to-bottom
jupyter notebook

# Or execute all at once (takes ~15 min on a modern laptop):
for nb in notebooks/0{1..9}*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

**4. Launch the Streamlit demo**
```bash
streamlit run app/streamlit_app.py
# → Opens at http://localhost:8501
```

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Gradient Boosting | LightGBM | 4.6.0 |
| Logistic Regression / WOE | scikit-learn | 1.9.0 |
| SHAP Explainability | SHAP | 0.52.0 |
| Fairness Auditing | Fairlearn | 0.13.0 |
| Statistical Testing | SciPy | 1.15.3 |
| Data Processing | Pandas / NumPy | 2.x / 1.x |
| Visualisation | Matplotlib / Seaborn | 3.x |
| Interactive Demo | Streamlit | 1.58.0 |
| Language | Python | 3.13 |
| Notebook Runtime | Jupyter / nbconvert | 1.x |

---

## Honest Limitations

- **Synthetic India layer** — the three borrower archetypes (kirana, gig worker, farmer)
  in `src/india_layer.py` and notebooks 06–07 are algorithmically generated from aggregate
  economic parameters. They are not derived from any real credit application data and should
  not be treated as representative of actual Indian borrower risk profiles.

- **Home Credit is not Indian data** — the model is trained on Home Credit's Eastern
  European loan portfolio. Feature distributions, default rates (~8%), and borrower
  demographics may differ significantly from Indian MSME or microfinance contexts.
  Transfer performance on real Indian data has not been validated.

- **No production validation** — TrustGraph has not undergone back-testing on out-of-time
  samples, champion/challenger testing, or regulatory review. It is a research and
  demonstration artefact. Do not use it to make real lending decisions.

---

## Citation

If you use TrustGraph in your research, please cite:

```bibtex
@misc{trustgraph2025,
  title        = {TrustGraph: Explainable Credit Risk Scoring for Thin-File Borrowers},
  author       = {Krish},
  year         = {2025},
  note         = {Working paper. Available at SSRN: https://ssrn.com/abstract=XXXXXXX},
  howpublished = {\url{https://github.com/<your-username>/TrustGraph}}
}
```

---

## Author

**Krish**
&nbsp;·&nbsp; [LinkedIn](#)
&nbsp;·&nbsp; [SSRN Paper](#)
&nbsp;·&nbsp; [GitHub](https://github.com/krish)

> *Built to demonstrate that credit invisibility is a data problem, not a creditworthiness problem.*
