"""
TrustGraph — Credit Risk Scoring Demo
Streamlit application for interactive borrower scoring.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_PROC  = ROOT / "data" / "processed"
SRC_DIR    = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="TrustGraph — Credit Risk Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — light, restrained, credit-report style ────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap');

:root {
    --ink:      #16212E;
    --muted:    #5B6675;
    --rule:     #DDE2E8;
    --panel:    #F6F8FA;
    --accent:   #23408E;
    --approve:  #1E7B4F;
    --refer:    #A86A00;
    --decline:  #B42318;
}

.stApp, .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th,
label, .stButton button p, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] p {
    font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
}
h1, h2, h3, h4 { font-family: 'IBM Plex Serif', Georgia, serif !important; color: var(--ink); font-weight: 600 !important; }
h1 { letter-spacing: -0.5px; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; color: var(--ink); }

/* Section header */
.section-header {
    font-family: 'IBM Plex Serif', Georgia, serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--ink);
    margin: 0.4rem 0 1rem 0;
}
.section-note { font-size: 0.9rem; color: var(--muted); margin: -0.6rem 0 1rem 0; }

/* Score card */
.score-card {
    background: #FFFFFF;
    border: 1px solid var(--rule);
    border-top: 4px solid var(--accent);
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin: 0.5rem 0;
}
.score-number {
    font-size: 3.6rem;
    font-weight: 600;
    line-height: 1;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
}
.score-number small { font-size: 1.1rem; color: var(--muted); font-weight: 400; }
.score-label { font-size: 0.9rem; color: var(--muted); margin-top: 0.35rem; }

/* Decision chip */
.badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 4px;
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 0.9rem;
}
.badge-approve { background: #E7F4EC; color: var(--approve); }
.badge-refer   { background: #FBF1DE; color: var(--refer); }
.badge-decline { background: #FBE9E7; color: var(--decline); }

/* Reason codes */
.reason-box {
    display: flex;
    gap: 0.75rem;
    align-items: baseline;
    border-bottom: 1px solid var(--rule);
    padding: 0.7rem 0;
    font-size: 0.97rem;
    color: var(--ink);
    line-height: 1.5;
}
.reason-sign { font-weight: 700; width: 1rem; flex-shrink: 0; text-align: center; }

/* Model comparison */
.model-col {
    background: var(--panel);
    border: 1px solid var(--rule);
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
}
.model-name { font-weight: 600; color: var(--ink); font-size: 1rem; }
.model-score { font-size: 2.6rem; font-weight: 600; color: var(--ink); margin: 0.4rem 0 0 0; font-variant-numeric: tabular-nums; }
.model-meta { font-size: 0.85rem; color: var(--muted); margin-top: 0.6rem; line-height: 1.5; }

/* Fairness note */
.fairness-pill {
    background: var(--panel);
    border-left: 3px solid var(--accent);
    padding: 0.9rem 1.2rem;
    font-size: 0.95rem;
    color: var(--ink);
    line-height: 1.6;
}

hr { border-color: var(--rule) !important; }
</style>
""", unsafe_allow_html=True)


# ── WOE helpers (needed at train time too) ─────────────────────────────────────
def _compute_woe_bins(series, target, n_bins=10):
    eps = 1e-6
    total_ev  = float(target.sum())
    total_nev = float(len(target) - total_ev)
    records   = []
    miss_mask = series.isna()
    if miss_mask.any():
        ev  = float(target[miss_mask].sum())
        nev = float(miss_mask.sum() - ev)
        d_ev = ev / total_ev + eps; d_nv = nev / total_nev + eps
        woe = np.log(d_ev / d_nv)
        records.append({"Bin": "MISSING", "Low": np.nan, "High": np.nan,
                        "Count": int(miss_mask.sum()), "Events": int(ev),
                        "NonEvents": int(nev), "EventRate": ev / max(miss_mask.sum(), 1),
                        "WoE": woe, "IV": (d_ev - d_nv) * woe})
    clean = series[~miss_mask].values
    tgt_c = target[~miss_mask].values
    bps   = np.unique(np.percentile(clean, np.linspace(0, 100, n_bins + 1)))
    for i in range(len(bps) - 1):
        lo, hi = bps[i], bps[i + 1]
        mask = (clean >= lo) & (clean <= hi) if i == len(bps) - 2 else (clean >= lo) & (clean < hi)
        if mask.sum() == 0:
            continue
        ev  = float(tgt_c[mask].sum())
        nev = float(mask.sum() - ev)
        d_ev = ev / total_ev + eps; d_nv = nev / total_nev + eps
        woe = np.log(d_ev / d_nv)
        records.append({"Bin": f"[{lo:.4g},{hi:.4g})", "Low": lo, "High": hi,
                        "Count": int(mask.sum()), "Events": int(ev),
                        "NonEvents": int(nev), "EventRate": ev / max(mask.sum(), 1),
                        "WoE": woe, "IV": (d_ev - d_nv) * woe})
    return pd.DataFrame(records)


def _fit_woe_pipeline(X, y, features, n_bins=10):
    pipeline = {}
    for feat in features:
        tbl  = _compute_woe_bins(X[feat], y, n_bins=n_bins)
        clean = X[feat].dropna().values
        bps  = np.unique(np.percentile(clean, np.linspace(0, 100, n_bins + 1))) if len(clean) else np.array([])
        pipeline[feat] = {"table": tbl, "breakpoints": bps, "iv": tbl["IV"].sum()}
    return pipeline


def _apply_woe_pipeline(X, pipeline):
    out = {}
    for feat, info in pipeline.items():
        tbl  = info["table"]
        bps  = info["breakpoints"]
        vals = X[feat].values.astype(float)
        result = np.zeros(len(vals))
        miss_r = tbl[tbl["Bin"] == "MISSING"]
        miss_woe = float(miss_r["WoE"].iloc[0]) if len(miss_r) else 0.0
        non_miss = tbl[tbl["Bin"] != "MISSING"].reset_index(drop=True)
        woe_arr  = non_miss["WoE"].values.astype(float)
        nan_mask = np.isnan(vals)
        result[nan_mask] = miss_woe
        if len(bps) >= 2 and len(woe_arr) > 0:
            bi = np.digitize(vals[~nan_mask], bps[1:], right=False)
            bi = np.clip(bi, 0, len(woe_arr) - 1)
            result[~nan_mask] = woe_arr[bi]
        out[f"{feat}_WOE"] = result
    return pd.DataFrame(out, index=X.index)


# ── Train models from processed data (used when no .pkl files exist) ───────────
def _train_lgbm(X_train, y_train, X_val, y_val):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def _train_scorecard(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    SC_FEATURES = [
        "EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "DAYS_EMPLOYED_RATIO",
        "AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "FLAG_MISSING_EXT",
    ]
    sc_feats = [f for f in SC_FEATURES if f in X_train.columns]
    woe_pl   = _fit_woe_pipeline(X_train[sc_feats], y_train, sc_feats, n_bins=10)
    X_woe    = _apply_woe_pipeline(X_train[sc_feats], woe_pl)
    lr = LogisticRegression(solver="lbfgs", max_iter=1000,
                            class_weight="balanced", random_state=42)
    lr.fit(X_woe, y_train)
    return {"lr_model": lr, "woe_pipeline": woe_pl, "features": sc_feats,
            "woe_feature_names": list(X_woe.columns)}


# ── Load or train models (cached for the full session) ─────────────────────────
@st.cache_resource(show_spinner="⏳ Setting up TrustGraph models — first run trains automatically (~2 min)…")
def load_models():
    """
    Load pre-trained models if .pkl files exist (local dev).
    Otherwise train from data/processed/features_train.csv (Streamlit Cloud).
    Models are cached for the entire session via st.cache_resource.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    lgbm_path = MODELS_DIR / "lgbm_final.pkl"
    sc_path   = MODELS_DIR / "scorecard_final.pkl"

    # ── Fast path: .pkl files present (local or if committed) ─────────────────
    if lgbm_path.exists() and sc_path.exists():
        with open(lgbm_path, "rb") as f:
            lgbm = pickle.load(f)
        with open(sc_path, "rb") as f:
            sc_bundle = pickle.load(f)
        return lgbm, sc_bundle

    # ── Slow path: train from features_train.csv or the tracked 15k sample ──────
    feat_path = DATA_PROC / "features_train.csv"
    if not feat_path.exists():
        feat_path = DATA_PROC / "features_sample.csv"   # 10 MB sample committed to git
    if not feat_path.exists():
        # Last resort: train from raw data if available, else raise clear error
        raw_path = ROOT / "data" / "raw" / "application_train.csv"
        if not raw_path.exists():
            st.error(
                "Neither model files nor processed data found. "
                "Please add data/processed/features_train.csv or run notebooks 01–03 first."
            )
            st.stop()

        # Minimal preprocessing from raw (mirrors notebook 03)
        df = pd.read_csv(raw_path)
        y  = df["TARGET"]
        X  = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

        # Engineered features
        X["CREDIT_INCOME_RATIO"]  = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"].replace(0, np.nan)
        X["ANNUITY_INCOME_RATIO"] = X["AMT_ANNUITY"] / X["AMT_INCOME_TOTAL"].replace(0, np.nan)
        X["CREDIT_TERM"]          = X["AMT_CREDIT"] / X["AMT_ANNUITY"].replace(0, np.nan)
        emp_clean = X["DAYS_EMPLOYED"].replace(365243, np.nan)
        X["DAYS_EMPLOYED_RATIO"]  = emp_clean / X["DAYS_BIRTH"].replace(0, np.nan)
        ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        X["EXT_SOURCE_MEAN"]      = X[ext_cols].mean(axis=1)
        X["EXT_SOURCE_MIN"]       = X[ext_cols].min(axis=1)
        X["INCOME_PER_PERSON"]    = X["AMT_INCOME_TOTAL"] / X["CNT_FAM_MEMBERS"].replace(0, np.nan)
        X["FLAG_MISSING_EXT"]     = X[ext_cols].isnull().any(axis=1).astype(int)

        le = LabelEncoder()
        for col in X.select_dtypes(include="object").columns:
            X[col] = le.fit_transform(X[col].fillna("MISSING").astype(str))
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(X[col].median())
    else:
        df = pd.read_csv(feat_path)
        y  = df["TARGET"]
        X  = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

    # Split (same params as notebooks)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Train both models
    lgbm      = _train_lgbm(X_train, y_train, X_val, y_val)
    sc_bundle = _train_scorecard(X_train, y_train)

    # Persist to models/ so subsequent cold starts skip retraining
    MODELS_DIR.mkdir(exist_ok=True)
    with open(lgbm_path, "wb") as f:
        pickle.dump(lgbm, f)
    with open(sc_path, "wb") as f:
        pickle.dump(sc_bundle, f)

    return lgbm, sc_bundle


@st.cache_data(show_spinner=False)
def load_fairness():
    path = DATA_PROC / "fairness_results.csv"
    if path.exists():
        df = pd.read_csv(path)
        dpd_row = df[df["Metric"] == "Demographic Parity Difference"]
        if not dpd_row.empty:
            pre  = dpd_row["Pre-Mitigation"].values[0]
            post = dpd_row["Post-Mitigation"].values[0]
            return float(pre), float(post)
    return None, None


lgbm_model, sc_bundle = load_models()
lgbm_features  = lgbm_model.feature_name_
sc_lr          = sc_bundle["lr_model"]
sc_woe_pipeline = sc_bundle["woe_pipeline"]
sc_features    = sc_bundle["features"]

dpd_pre, dpd_post = load_fairness()


# ── Helper: WOE transform (vectorised) ────────────────────────────────────────
def woe_transform_series(vals_arr, woe_table, breakpoints):
    result   = np.zeros(len(vals_arr))
    miss_r   = woe_table[woe_table["Bin"] == "MISSING"]
    miss_woe = float(miss_r["WoE"].iloc[0]) if len(miss_r) else 0.0
    non_miss = woe_table[woe_table["Bin"] != "MISSING"].reset_index(drop=True)
    woe_arr  = non_miss["WoE"].values.astype(float)
    nan_mask = np.isnan(vals_arr)
    result[nan_mask] = miss_woe
    if len(breakpoints) >= 2 and len(woe_arr) > 0:
        bi = np.digitize(vals_arr[~nan_mask].astype(float), breakpoints[1:], right=False)
        bi = np.clip(bi, 0, len(woe_arr) - 1)
        result[~nan_mask] = woe_arr[bi]
    return result


def apply_woe_single_row(row_dict, pipeline, features):
    out = {}
    for feat in features:
        info = pipeline[feat]
        val  = row_dict.get(feat, np.nan)
        arr  = np.array([val], dtype=float)
        out[f"{feat}_WOE"] = woe_transform_series(arr, info["table"], info["breakpoints"])[0]
    return out


# ── Helper: score conversion ───────────────────────────────────────────────────
def prob_to_score(prob: float) -> int:
    """Convert default probability to a 0-1000 credit score (higher = safer)."""
    return int(round((1 - prob) * 1000))


def score_to_decision(score: int):
    if score >= 600:
        return "Approve",  "badge-approve",  ""
    elif score >= 400:
        return "Refer",    "badge-refer",    ""
    else:
        return "Decline",  "badge-decline",  ""


# ── Helper: plain-English reason codes ────────────────────────────────────────
FEATURE_SIGNALS = {
    "EXT_SOURCE_MEAN":       ("strong bureau score",          "weak / missing bureau score"),
    "EXT_SOURCE_MIN":        ("consistent bureau ratings",    "at least one very low bureau score"),
    "CREDIT_INCOME_RATIO":   ("manageable loan-to-income",    "loan amount high relative to income"),
    "ANNUITY_INCOME_RATIO":  ("affordable repayment burden",  "debt obligations exceed safe threshold"),
    "CREDIT_TERM":           ("standard loan tenure",         "unusually long repayment period"),
    "DAYS_EMPLOYED_RATIO":   ("stable employment history",    "limited employment tenure"),
    "DAYS_BIRTH":            ("experienced borrower profile", "applicant age outside typical band"),
    "AMT_INCOME_TOTAL":      ("solid annual income",          "lower annual income"),
    "AMT_CREDIT":            ("loan amount within norms",     "high loan amount requested"),
    "FLAG_MISSING_EXT":      ("bureau data available",        "thin-file — no formal credit history"),
    "INCOME_PER_PERSON":     ("good per-capita income",       "income stretched across many dependants"),
}


def generate_reason_codes(row_dict: dict, prob: float, top_n: int = 3) -> list:
    """
    Compute a lightweight feature-contribution signal and return plain-English lines.
    Uses the sign and magnitude of key engineered features as a SHAP-like proxy.
    """
    # Compute signed contributions (deviation from neutral)
    contributions = {}

    ext_mean = row_dict.get("EXT_SOURCE_MEAN", np.nan)
    if not np.isnan(ext_mean):
        contributions["EXT_SOURCE_MEAN"] = ext_mean - 0.5          # > 0 = good

    cir = row_dict.get("CREDIT_INCOME_RATIO", np.nan)
    if not np.isnan(cir):
        contributions["CREDIT_INCOME_RATIO"] = -(cir - 3.0) / 3.0  # high ratio = bad

    air = row_dict.get("ANNUITY_INCOME_RATIO", np.nan)
    if not np.isnan(air):
        contributions["ANNUITY_INCOME_RATIO"] = -(air - 0.15) / 0.15

    der = row_dict.get("DAYS_EMPLOYED_RATIO", np.nan)
    if not np.isnan(der):
        contributions["DAYS_EMPLOYED_RATIO"] = der - 0.3            # high = good

    flag_miss = row_dict.get("FLAG_MISSING_EXT", 0)
    contributions["FLAG_MISSING_EXT"] = -1.2 if flag_miss else 0.8

    credit_term = row_dict.get("CREDIT_TERM", np.nan)
    if not np.isnan(credit_term):
        contributions["CREDIT_TERM"] = -(credit_term - 36) / 36

    income = row_dict.get("AMT_INCOME_TOTAL", np.nan)
    if not np.isnan(income):
        contributions["AMT_INCOME_TOTAL"] = (income - 200_000) / 200_000

    # Sort by absolute contribution
    sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    reasons = []
    for feat, contrib in sorted_feats[:top_n]:
        positive = contrib > 0
        label    = FEATURE_SIGNALS.get(feat, ("positive signal", "risk signal"))
        text     = label[0] if positive else label[1]
        icon     = "+" if positive else "−"
        reasons.append((icon, text, positive))

    return reasons


# ── Helper: build feature vector for LightGBM ─────────────────────────────────
def build_lgbm_row(inputs: dict) -> pd.DataFrame:
    """Map UI inputs to the full 128-feature vector the model expects."""
    defaults = {f: 0.0 for f in lgbm_features}

    # Engineered features
    defaults["AMT_CREDIT"]           = inputs["loan_amount"]
    defaults["AMT_ANNUITY"]          = inputs["loan_amount"] / max(inputs["loan_term"], 1)
    defaults["AMT_INCOME_TOTAL"]     = inputs["monthly_income"] * 12
    defaults["DAYS_BIRTH"]           = inputs.get("days_birth", -12000)
    defaults["DAYS_EMPLOYED"]        = -abs(inputs["days_employed"])
    defaults["CNT_FAM_MEMBERS"]      = inputs["family_members"]

    ann_income  = defaults["AMT_ANNUITY"]
    tot_income  = defaults["AMT_INCOME_TOTAL"]
    days_birth  = defaults["DAYS_BIRTH"]
    days_emp    = defaults["DAYS_EMPLOYED"]

    defaults["CREDIT_INCOME_RATIO"]  = inputs["loan_amount"] / max(tot_income, 1)
    defaults["ANNUITY_INCOME_RATIO"] = ann_income / max(inputs["monthly_income"], 1)
    defaults["CREDIT_TERM"]          = inputs["loan_amount"] / max(ann_income, 1)
    defaults["DAYS_EMPLOYED_RATIO"]  = days_emp / days_birth if days_birth != 0 else 0.0
    defaults["EXT_SOURCE_MEAN"]      = np.nan if inputs["flag_missing_ext"] else inputs["ext_source_mean"]
    defaults["EXT_SOURCE_MIN"]       = np.nan if inputs["flag_missing_ext"] else inputs["ext_source_mean"] * 0.85
    defaults["INCOME_PER_PERSON"]    = tot_income / max(inputs["family_members"], 1)
    defaults["FLAG_MISSING_EXT"]     = 1 if inputs["flag_missing_ext"] else 0

    row = {f: defaults.get(f, 0.0) for f in lgbm_features}
    return pd.DataFrame([row])


def build_sc_row(inputs: dict) -> dict:
    """Build scorecard input dict."""
    ann_income = inputs["loan_amount"] / max(inputs["loan_term"], 1)
    tot_income = inputs["monthly_income"] * 12
    days_birth = inputs.get("days_birth", -12000)
    days_emp   = -abs(inputs["days_employed"])
    return {
        "EXT_SOURCE_MEAN":      np.nan if inputs["flag_missing_ext"] else inputs["ext_source_mean"],
        "EXT_SOURCE_MIN":       np.nan if inputs["flag_missing_ext"] else inputs["ext_source_mean"] * 0.85,
        "CREDIT_INCOME_RATIO":  inputs["loan_amount"] / max(tot_income, 1),
        "ANNUITY_INCOME_RATIO": ann_income / max(inputs["monthly_income"], 1),
        "CREDIT_TERM":          inputs["loan_amount"] / max(ann_income, 1),
        "DAYS_EMPLOYED_RATIO":  days_emp / days_birth if days_birth != 0 else 0.0,
        "AMT_CREDIT":           inputs["loan_amount"],
        "AMT_INCOME_TOTAL":     tot_income,
        "DAYS_BIRTH":           days_birth,
        "FLAG_MISSING_EXT":     1 if inputs["flag_missing_ext"] else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## TrustGraph")
    st.markdown("""
    **TrustGraph** is an end-to-end credit risk scoring engine built for thin-file
    borrowers in emerging markets. It combines gradient-boosted models, WOE scorecards,
    SHAP explainability, and Fairlearn bias mitigation into a single auditable pipeline.
    """)

    st.markdown("---")
    st.markdown("[View the code on GitHub](https://github.com/krishmakhija13/-TrustGraph)")

    st.markdown("---")
    st.markdown("#### Tech stack")
    st.markdown("""
    | Component | Library |
    |---|---|
    | Gradient Boosting | LightGBM |
    | WOE Scorecard | Custom (sklearn) |
    | Explainability | SHAP |
    | Fairness Audit | Fairlearn |
    | UI | Streamlit |
    | Language | Python 3.13 |
    """)

    st.markdown("---")
    st.markdown("#### Disclaimer")
    st.caption(
        "Demo only. All India profiles are synthetic. "
        "Model trained on Home Credit Open Dataset (Kaggle). "
        "Not for production lending decisions."
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# TrustGraph credit scoring")
st.markdown("Score a borrower, see the reasons behind the decision, and compare an accurate model with an explainable one.")
st.markdown("---")

# ── Quick-fill presets ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Start with a sample borrower</div>', unsafe_allow_html=True)
col_p1, col_p2, col_p3, col_p4 = st.columns(4)

preset = None
with col_p1:
    if st.button("Kirana owner", use_container_width=True):
        preset = "kirana"
with col_p2:
    if st.button("Gig worker", use_container_width=True):
        preset = "gig"
with col_p3:
    if st.button("Small farmer", use_container_width=True):
        preset = "farmer"
with col_p4:
    if st.button("Salaried (prime)", use_container_width=True):
        preset = "salaried"

PRESETS = {
    "kirana":   dict(income_type="Self-employed / Kirana", monthly_income=42000, loan_amount=200000, loan_term=36,  ext_score=0.45, days_emp=2500, family=4, thin_file=True),
    "gig":      dict(income_type="Gig Worker",             monthly_income=25000, loan_amount=100000, loan_term=24,  ext_score=0.40, days_emp=1200, family=2, thin_file=True),
    "farmer":   dict(income_type="Farmer",                 monthly_income=20000, loan_amount=150000, loan_term=48,  ext_score=0.35, days_emp=5000, family=5, thin_file=True),
    "salaried": dict(income_type="Salaried",               monthly_income=80000, loan_amount=500000, loan_term=60,  ext_score=0.72, days_emp=3000, family=3, thin_file=False),
}

# Initialise session state
if "inputs" not in st.session_state:
    st.session_state.inputs = PRESETS["salaried"]
if preset:
    st.session_state.inputs = PRESETS[preset]

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BORROWER PROFILE INPUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Borrower profile</div>', unsafe_allow_html=True)

inp = st.session_state.inputs

col_a, col_b = st.columns([1, 1], gap="large")

with col_a:
    income_type = st.selectbox(
        "Income type",
        ["Salaried", "Self-employed / Kirana", "Gig Worker", "Farmer"],
        index=["Salaried", "Self-employed / Kirana", "Gig Worker", "Farmer"].index(inp["income_type"])
        if inp["income_type"] in ["Salaried", "Self-employed / Kirana", "Gig Worker", "Farmer"] else 0,
        help="Borrower's primary source of income"
    )

    monthly_income = st.slider(
        "Monthly income (₹)",
        min_value=5_000,
        max_value=200_000,
        value=inp["monthly_income"],
        step=1_000,
        format="₹%d",
        help="Average monthly take-home income in INR"
    )

    loan_amount = st.slider(
        "Loan amount requested (₹)",
        min_value=10_000,
        max_value=2_000_000,
        value=inp["loan_amount"],
        step=5_000,
        format="₹%d",
        help="Principal loan amount requested"
    )

    loan_term = st.slider(
        "Loan term (months)",
        min_value=12,
        max_value=120,
        value=inp["loan_term"],
        step=6,
        help="Repayment period in months"
    )

with col_b:
    ext_source_mean = st.slider(
        "Credit bureau score  (0.0 = no history, 1.0 = excellent)",
        min_value=0.0,
        max_value=1.0,
        value=float(inp["ext_score"]),
        step=0.01,
        help="Average external credit bureau score across all bureaus"
    )

    days_employed = st.number_input(
        "Days at current employer",
        min_value=0,
        max_value=20_000,
        value=int(inp["days_emp"]),
        step=100,
        help="Number of days employed at current job"
    )

    family_members = st.number_input(
        "Family members (incl. applicant)",
        min_value=1,
        max_value=15,
        value=int(inp["family"]),
        step=1,
        help="Total household size including the applicant"
    )

    flag_missing_ext = st.checkbox(
        "No formal credit history (thin-file applicant)",
        value=bool(inp["thin_file"]),
        help="Check if the borrower has no credit bureau record"
    )

# Derived display metrics
annuity      = loan_amount / max(loan_term, 1)
annual_income = monthly_income * 12
cir          = loan_amount / max(annual_income, 1)
air          = annuity / max(monthly_income, 1)

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Annual income", f"₹{annual_income:,.0f}")
mc2.metric("Monthly EMI", f"₹{annuity:,.0f}")
mc3.metric("Loan-to-Income", f"{cir:.1f}×", delta=f"{'High' if cir > 5 else 'OK'}", delta_color="inverse" if cir > 5 else "normal")
mc4.metric("EMI to income", f"{air:.1%}", delta=f"{'Stretched' if air > 0.5 else 'OK'}", delta_color="inverse" if air > 0.5 else "normal")

# Build input dict for scoring
days_birth_approx = -12000  # ~33 years old default
inputs = dict(
    income_type=income_type,
    monthly_income=monthly_income,
    loan_amount=loan_amount,
    loan_term=loan_term,
    ext_source_mean=ext_source_mean,
    days_employed=days_employed,
    family_members=family_members,
    flag_missing_ext=flag_missing_ext,
    days_birth=days_birth_approx,
)

# ── Compute scores ─────────────────────────────────────────────────────────────
X_lgbm   = build_lgbm_row(inputs)
lgbm_prob = float(lgbm_model.predict_proba(X_lgbm)[0, 1])
lgbm_score = prob_to_score(lgbm_prob)

sc_row_dict = build_sc_row(inputs)
sc_woe_row  = apply_woe_single_row(sc_row_dict, sc_woe_pipeline, sc_features)
sc_woe_df   = pd.DataFrame([sc_woe_row])
sc_prob     = float(sc_lr.predict_proba(sc_woe_df)[0, 1])
sc_score    = prob_to_score(sc_prob)

decision, badge_class, dec_icon = score_to_decision(lgbm_score)
reason_codes = generate_reason_codes({**inputs, **sc_row_dict}, lgbm_prob)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TRUSTGRAPH SCORE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">TrustGraph score</div>', unsafe_allow_html=True)

score_col, reason_col = st.columns([1, 2], gap="large")

with score_col:
    score_color = {"Approve": "#1E7B4F", "Refer": "#A86A00", "Decline": "#B42318"}[decision]

    st.markdown(f"""
    <div class="score-card">
        <div class="score-number">{lgbm_score}<small> / 1000</small></div>
        <div class="score-label">TrustGraph score</div>
        <span class="badge {badge_class}">{decision}</span>
        <div style="margin-top:0.9rem; font-size:0.85rem; color:#5B6675;">
            Probability of default: {lgbm_prob:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.85rem; color:#5B6675; margin-top:0.8rem; line-height:1.8;">
    600 to 1000: approve<br>
    400 to 599: refer for manual review<br>
    Below 400: decline
    </div>
    """, unsafe_allow_html=True)

with reason_col:
    st.markdown("**What drove this decision**")
    for icon, text, positive in reason_codes:
        colour = "#1E7B4F" if positive else "#B42318"
        st.markdown(
            f'<div class="reason-box">'
            f'<span class="reason-sign" style="color:{colour};">{icon}</span>'
            f'<span>{text.capitalize()}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if flag_missing_ext:
        st.markdown(
            '<div class="reason-box">'
            '<span class="reason-sign" style="color:#A86A00;">!</span>'
            '<span>Thin-file applicant with no credit bureau record. '
            'The score relies on income and loan structure alone.</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Score progress bar
    st.markdown(f"""
    <div style="margin-top:1rem;">
        <div style="font-size:0.85rem; color:#5B6675; margin-bottom:0.3rem;">Where this score sits</div>
        <div style="background:#E6EAEF; border-radius:999px; height:8px; overflow:hidden;">
            <div style="width:{lgbm_score/10:.0f}%; background:{score_color}; height:100%; border-radius:999px;
                        transition: width 0.5s ease;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#5B6675; margin-top:0.2rem;">
            <span>0</span><span>Decline</span><span>Refer</span><span>Approve</span><span>1000</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Two models, same borrower</div>', unsafe_allow_html=True)

lgbm_dec, lgbm_badge, lgbm_icon = score_to_decision(lgbm_score)
sc_dec,   sc_badge,   sc_icon   = score_to_decision(sc_score)

cmp_col1, cmp_spacer, cmp_col2 = st.columns([5, 1, 5])

with cmp_col1:
    st.markdown(f"""
    <div class="model-col">
        <div class="model-name">LightGBM</div>
        <div class="model-score">{lgbm_score}</div>
        <span class="badge {lgbm_badge}">{lgbm_dec}</span>
        <div class="model-meta">Probability of default: {lgbm_prob:.1%}<br>Built for accuracy: 128 features, early stopping</div>
    </div>
    """, unsafe_allow_html=True)

with cmp_col2:
    st.markdown(f"""
    <div class="model-col">
        <div class="model-name">WOE scorecard</div>
        <div class="model-score">{sc_score}</div>
        <span class="badge {sc_badge}">{sc_dec}</span>
        <div class="model-meta">Probability of default: {sc_prob:.1%}<br>Built for transparency: 10 features, auditable points table</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:1rem; font-size:0.92rem; color:#5B6675; max-width:70ch;">
    LightGBM is more accurate; the scorecard is easier to explain. Use LightGBM for portfolio-level decisions,
    and the scorecard when a loan officer must explain the outcome to the applicant.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FAIRNESS NOTE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Fairness audit</div>', unsafe_allow_html=True)

if dpd_pre is not None:
    dpd_status = "within acceptable range" if abs(float(dpd_post)) <= 0.05 else "mitigated via ThresholdOptimizer"
    st.markdown(f"""
    <div class="fairness-pill">
        This model was audited for gender bias with Fairlearn.
        Demographic parity difference fell from <b>{dpd_pre:.4f}</b> before mitigation to <b>{dpd_post:.4f}</b> after ({dpd_status}),
        equalising approval rates between men and women.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="fairness-pill">
        This model was audited for gender bias with Fairlearn.
        Run notebook 08_fairness.ipynb to generate fairness metrics.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="font-size:0.85rem; color:#5B6675; margin-top:0.6rem; max-width:80ch;">
    Fairness audit scope: gender parity (CODE_GENDER proxy), demographic parity difference,
    equalized odds difference. Mitigation method: post-processing ThresholdOptimizer.
    Audit cadence: recommended monthly in production.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("TrustGraph v1.0. Trained on the Home Credit open dataset. For research and demonstration only.")
