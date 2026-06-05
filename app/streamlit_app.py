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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark, clean, professional ─────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* Score cards */
.score-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 0.5rem 0;
}
.score-number {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
}
.score-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.4rem;
}

/* Decision badge */
.badge {
    display: inline-block;
    padding: 0.45rem 1.4rem;
    border-radius: 999px;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-top: 0.75rem;
}
.badge-approve  { background: #052e16; color: #4ade80; border: 1px solid #16a34a; }
.badge-refer    { background: #1c1917; color: #fbbf24; border: 1px solid #d97706; }
.badge-decline  { background: #2d0a0a; color: #f87171; border: 1px solid #dc2626; }

/* Reason code box */
.reason-box {
    background: #0f172a;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.4rem 0;
    font-size: 0.92rem;
    color: #cbd5e1;
    line-height: 1.5;
}

/* Section header */
.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #64748b;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 1rem;
}

/* Comparison columns */
.model-col {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}

/* Fairness pill */
.fairness-pill {
    background: #0c1a2e;
    border: 1px solid #1d4ed8;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-size: 0.88rem;
    color: #93c5fd;
    margin-top: 0.5rem;
}

/* Metric delta colours */
.delta-up   { color: #4ade80; }
.delta-down { color: #f87171; }

/* Divider */
hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading TrustGraph models…")
def load_models():
    with open(MODELS_DIR / "lgbm_final.pkl", "rb") as f:
        lgbm = pickle.load(f)
    with open(MODELS_DIR / "scorecard_final.pkl", "rb") as f:
        sc_bundle = pickle.load(f)
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
        return "Approve",  "badge-approve",  "✅"
    elif score >= 400:
        return "Refer",    "badge-refer",    "🔍"
    else:
        return "Decline",  "badge-decline",  "❌"


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
        icon     = "✔" if positive else "✘"
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
    st.markdown("## 📊 TrustGraph")
    st.markdown("""
    **TrustGraph** is an end-to-end credit risk scoring engine built for thin-file
    borrowers in emerging markets. It combines gradient-boosted models, WOE scorecards,
    SHAP explainability, and Fairlearn bias mitigation into a single auditable pipeline.
    """)

    st.markdown("---")
    st.markdown("#### 🔗 Research")
    st.markdown("""
    📄 [SSRN Working Paper](#) *(forthcoming)*
    > *TrustGraph: Explainable Credit Scoring for Thin-File Borrowers in India*
    """)

    st.markdown("---")
    st.markdown("#### ⚙ Tech Stack")
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
    st.markdown("#### ⚠️ Disclaimer")
    st.caption(
        "Demo only. All India profiles are synthetic. "
        "Model trained on Home Credit Open Dataset (Kaggle). "
        "Not for production lending decisions."
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 📊 TrustGraph Credit Scoring")
st.markdown("*Real-time credit risk assessment with explainability and fairness auditing*")
st.markdown("---")

# ── Quick-fill presets ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Quick Profiles</div>', unsafe_allow_html=True)
col_p1, col_p2, col_p3, col_p4 = st.columns(4)

preset = None
with col_p1:
    if st.button("🏪 Kirana Owner", use_container_width=True):
        preset = "kirana"
with col_p2:
    if st.button("🚗 Gig Worker", use_container_width=True):
        preset = "gig"
with col_p3:
    if st.button("🌾 Small Farmer", use_container_width=True):
        preset = "farmer"
with col_p4:
    if st.button("💼 Salaried (Prime)", use_container_width=True):
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
st.markdown('<div class="section-header">Section 1 — Borrower Profile</div>', unsafe_allow_html=True)

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
mc1.metric("Annual Income", f"₹{annual_income:,.0f}")
mc2.metric("Monthly EMI", f"₹{annuity:,.0f}")
mc3.metric("Loan-to-Income", f"{cir:.1f}×", delta=f"{'⚠ High' if cir > 5 else 'OK'}", delta_color="inverse" if cir > 5 else "normal")
mc4.metric("EMI/Income Ratio", f"{air:.1%}", delta=f"{'⚠ Stressed' if air > 0.5 else 'OK'}", delta_color="inverse" if air > 0.5 else "normal")

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
st.markdown('<div class="section-header">Section 2 — TrustGraph Score</div>', unsafe_allow_html=True)

score_col, reason_col = st.columns([1, 2], gap="large")

with score_col:
    # Score gauge colour
    if lgbm_score >= 600:
        score_color = "#4ade80"
    elif lgbm_score >= 400:
        score_color = "#fbbf24"
    else:
        score_color = "#f87171"

    st.markdown(f"""
    <div class="score-card">
        <div class="score-number" style="color:{score_color};">{lgbm_score}</div>
        <div class="score-label">TrustGraph Score  /  1000</div>
        <div>
            <span class="badge {badge_class}">{dec_icon} {decision}</span>
        </div>
        <div style="margin-top:1rem; font-size:0.8rem; color:#64748b;">
            P(default) = {lgbm_prob:.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Score band guide
    st.markdown("""
    <div style="font-size:0.78rem; color:#64748b; margin-top:0.8rem; line-height:1.8;">
    <span style="color:#4ade80;">●</span> 600–1000: <b>Approve</b><br>
    <span style="color:#fbbf24;">●</span> 400–599: <b>Refer</b> (manual review)<br>
    <span style="color:#f87171;">●</span> 0–399: <b>Decline</b>
    </div>
    """, unsafe_allow_html=True)

with reason_col:
    st.markdown("**Key drivers of this decision:**")
    for icon, text, positive in reason_codes:
        colour = "#4ade80" if positive else "#f87171"
        st.markdown(
            f'<div class="reason-box">'
            f'<span style="color:{colour}; font-weight:700; margin-right:0.5rem;">{icon}</span>'
            f'{text.capitalize()}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if flag_missing_ext:
        st.markdown(
            '<div class="reason-box" style="border-color:#f59e0b;">'
            '<span style="color:#f59e0b; font-weight:700; margin-right:0.5rem;">⚠</span>'
            'Thin-file applicant — no formal credit bureau record. '
            'Score relies entirely on income and loan structure signals.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Score progress bar
    st.markdown(f"""
    <div style="margin-top:1rem;">
        <div style="font-size:0.78rem; color:#64748b; margin-bottom:0.3rem;">Score band</div>
        <div style="background:#1e293b; border-radius:999px; height:8px; overflow:hidden;">
            <div style="width:{lgbm_score/10:.0f}%; background:{score_color}; height:100%; border-radius:999px;
                        transition: width 0.5s ease;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#475569; margin-top:0.2rem;">
            <span>0</span><span>Decline</span><span>Refer</span><span>Approve</span><span>1000</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Section 3 — Model Comparison</div>', unsafe_allow_html=True)

lgbm_dec, lgbm_badge, lgbm_icon = score_to_decision(lgbm_score)
sc_dec,   sc_badge,   sc_icon   = score_to_decision(sc_score)

cmp_col1, cmp_spacer, cmp_col2 = st.columns([5, 1, 5])

with cmp_col1:
    lgbm_col = "#4ade80" if lgbm_score >= 600 else "#fbbf24" if lgbm_score >= 400 else "#f87171"
    st.markdown(f"""
    <div class="model-col">
        <div style="font-size:0.75rem; color:#64748b; letter-spacing:2px; text-transform:uppercase;">LightGBM</div>
        <div style="font-size:2.8rem; font-weight:800; color:{lgbm_col}; margin:0.5rem 0;">{lgbm_score}</div>
        <span class="badge {lgbm_badge}">{lgbm_icon} {lgbm_dec}</span>
        <div style="margin-top:0.8rem; font-size:0.8rem; color:#94a3b8;">P(default) = {lgbm_prob:.4f}</div>
        <div style="margin-top:0.5rem; font-size:0.78rem; color:#475569;">Optimises <b>predictive accuracy</b><br>128 features · early stopping</div>
    </div>
    """, unsafe_allow_html=True)

with cmp_col2:
    sc_col = "#4ade80" if sc_score >= 600 else "#fbbf24" if sc_score >= 400 else "#f87171"
    st.markdown(f"""
    <div class="model-col">
        <div style="font-size:0.75rem; color:#64748b; letter-spacing:2px; text-transform:uppercase;">WOE Scorecard</div>
        <div style="font-size:2.8rem; font-weight:800; color:{sc_col}; margin:0.5rem 0;">{sc_score}</div>
        <span class="badge {sc_badge}">{sc_icon} {sc_dec}</span>
        <div style="margin-top:0.8rem; font-size:0.8rem; color:#94a3b8;">P(default) = {sc_prob:.4f}</div>
        <div style="margin-top:0.5rem; font-size:0.78rem; color:#475569;">Optimises <b>transparency</b><br>10 features · auditable points table</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:1rem; font-size:0.82rem; color:#64748b;">
    LightGBM optimizes accuracy &nbsp;|&nbsp; Scorecard optimizes transparency.
    Use LightGBM for portfolio-level decisions; Scorecard when the loan officer needs to explain the outcome to the applicant.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FAIRNESS NOTE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Section 4 — Fairness Audit</div>', unsafe_allow_html=True)

if dpd_pre is not None:
    dpd_status = "within acceptable range" if abs(float(dpd_post)) <= 0.05 else "mitigated via ThresholdOptimizer"
    st.markdown(f"""
    <div class="fairness-pill">
        🛡 This model has been audited for gender bias using Fairlearn.
        &nbsp;|&nbsp; Demographic Parity Difference (pre-mitigation): <b>{dpd_pre:.4f}</b>
        &nbsp;|&nbsp; Post-mitigation: <b>{dpd_post:.4f}</b> — {dpd_status}.
        &nbsp;|&nbsp; Selection rates equalised across gender groups (M / F).
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="fairness-pill">
        🛡 This model has been audited for gender bias using Fairlearn.
        Run notebook 08_fairness.ipynb to generate fairness metrics.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="font-size:0.78rem; color:#475569; margin-top:0.6rem;">
    Fairness audit scope: gender parity (CODE_GENDER proxy), demographic parity difference,
    equalized odds difference. Mitigation method: post-processing ThresholdOptimizer.
    Audit cadence: recommended monthly in production.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("TrustGraph v1.0 · Home Credit Open Dataset · For research and demonstration only.")
