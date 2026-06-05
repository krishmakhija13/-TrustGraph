"""
india_layer.py — Synthetic Indian borrower cash-flow generator.

IMPORTANT: All data in this module is SYNTHETIC and SIMULATED.
It is generated programmatically for research and model-testing purposes only.
It does NOT represent any real individual, transaction, or credit record.
Economic parameters are loosely calibrated to publicly available RBI / MSME
aggregate statistics but are NOT precise reproductions of those statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

RNG = np.random.default_rng(seed=42)

# ── Monthly income profiles (INR) ────────────────────────────────────────────

def _kirana_monthly_income(months: int = 12) -> List[float]:
    """
    Generate synthetic monthly income for a kirana (neighbourhood grocery) shop owner.

    Seasonal pattern: dip during monsoon (Jun–Aug), peak around Diwali (Oct–Nov).
    Base range: INR 25,000–60,000 per month.

    SYNTHETIC DATA — not derived from any real borrower or business record.
    """
    base = 42000.0
    seasonal = np.array([
        1.00, 1.02, 1.05, 1.03, 1.01,  # Jan–May mild growth
        0.82, 0.78, 0.80,               # Jun–Aug monsoon dip
        0.95, 1.18, 1.22,               # Sep–Nov Diwali build-up & peak
        1.08,                           # Dec post-festival normalisation
    ])
    noise = RNG.normal(1.0, 0.05, months)
    incomes = base * seasonal[:months] * noise
    return np.clip(incomes, 25_000, 60_000).tolist()


def _gig_monthly_income(months: int = 12) -> List[float]:
    """
    Generate synthetic monthly income for a gig worker (food-delivery / ride-hailing).

    Income is irregular week-to-week; monthly totals in INR 15,000–35,000.
    Models high variance typical of platform-economy workers.

    SYNTHETIC DATA — not derived from any real borrower or transaction record.
    """
    base = 24000.0
    noise = RNG.normal(1.0, 0.15, months)   # high variance
    incomes = base * noise
    return np.clip(incomes, 15_000, 35_000).tolist()


def _farmer_monthly_income(months: int = 12) -> List[float]:
    """
    Generate synthetic monthly income for a small farmer (2 harvests/year).

    Income arrives in large lumps at Rabi (Mar) and Kharif (Oct) harvest seasons;
    near-zero in other months. Monthly average INR 12,000–28,000 when annualised.

    SYNTHETIC DATA — not derived from any real borrower or farm record.
    """
    annual_income = RNG.uniform(144_000, 336_000)   # 12k–28k monthly average
    monthly = np.zeros(months)
    # Kharif harvest: October = index 9; Rabi harvest: March = index 2
    for i in range(months):
        month_idx = i % 12
        if month_idx == 9:    # October — Kharif
            monthly[i] = annual_income * 0.55 + RNG.normal(0, 8_000)
        elif month_idx == 2:  # March — Rabi
            monthly[i] = annual_income * 0.45 + RNG.normal(0, 6_000)
        else:
            monthly[i] = RNG.uniform(500, 2_000)   # subsistence / odd jobs
    return np.clip(monthly, 0, None).tolist()


# ── Transaction generator ─────────────────────────────────────────────────────

def generate_transactions(profile_type: str, months: int = 12) -> pd.DataFrame:
    """
    Simulate monthly cash-flow transactions for a given synthetic borrower profile.

    Parameters
    ----------
    profile_type : str
        One of 'kirana', 'gig', 'farmer'.
    months : int
        Number of months to simulate (default 12).

    Returns
    -------
    pd.DataFrame
        Month-by-month income, estimated expenses, and net cash flow.

    SYNTHETIC DATA — all values are algorithmically generated and do NOT
    represent any real individual or business.
    """
    generators = {
        'kirana': _kirana_monthly_income,
        'gig':    _gig_monthly_income,
        'farmer': _farmer_monthly_income,
    }
    if profile_type not in generators:
        raise ValueError(f"Unknown profile_type '{profile_type}'. Choose from {list(generators)}")

    incomes   = generators[profile_type](months)
    # Expenses: rent + food + misc proportional to income, with floor
    expenses  = [max(8_000, inc * RNG.uniform(0.45, 0.65)) for inc in incomes]
    net_flows = [inc - exp for inc, exp in zip(incomes, expenses)]

    months_list = pd.date_range('2023-01', periods=months, freq='MS')
    return pd.DataFrame({
        'month':          months_list,
        'income_inr':     np.round(incomes, 2),
        'expenses_inr':   np.round(expenses, 2),
        'net_cashflow_inr': np.round(net_flows, 2),
    })


# ── Feature mapping to TrustGraph schema ─────────────────────────────────────

# Nominal TrustGraph feature medians used as fill-in for schema columns that
# cannot be derived from synthetic cash-flow alone.  These are dataset medians,
# NOT individual estimates.
_SCHEMA_DEFAULTS = {
    'AMT_GOODS_PRICE':          450_000.0,
    'DAYS_ID_PUBLISH':          -2_000.0,
    'DAYS_REGISTRATION':        -4_000.0,
    'REGION_POPULATION_RELATIVE': 0.02,
    'REGION_RATING_CLIENT':     2.0,
    'FLAG_OWN_CAR':             0,
    'FLAG_OWN_REALTY':          1,
    'CNT_CHILDREN':             0,
    'EXT_SOURCE_1':             np.nan,
    'EXT_SOURCE_2':             np.nan,
    'EXT_SOURCE_3':             np.nan,
}


def map_to_trustgraph_schema(
    txn_df: pd.DataFrame,
    profile_type: str,
    loan_amount_inr: float = 150_000.0,
    loan_term_months: int = 36,
) -> Dict:
    """
    Map a synthetic cash-flow DataFrame to the TrustGraph model feature schema.

    Converts rupee amounts to the scale used in the Home Credit dataset
    (treated as a common currency unit — purely for model compatibility).
    All computed features use the same definitions as 03_features.ipynb.

    Parameters
    ----------
    txn_df : pd.DataFrame
        Output of generate_transactions().
    profile_type : str
        Borrower type label (used to set profile-specific defaults).
    loan_amount_inr : float
        Requested loan amount in INR.
    loan_term_months : int
        Loan tenor in months.

    Returns
    -------
    dict
        Single-row feature dict compatible with the TrustGraph feature schema.

    SYNTHETIC DATA — all returned values are derived from simulated cash-flows
    and do NOT represent any real credit application.
    """
    avg_income = txn_df['income_inr'].mean()
    income_std = txn_df['income_inr'].std()
    min_income = txn_df['income_inr'].min()

    annuity = loan_amount_inr / loan_term_months

    # Age and employment assumptions per profile (synthetic)
    profile_defaults = {
        'kirana': {'DAYS_BIRTH': -12_000, 'DAYS_EMPLOYED': -2_500, 'CNT_FAM_MEMBERS': 4.0},
        'gig':    {'DAYS_BIRTH': -10_000, 'DAYS_EMPLOYED': -1_200, 'CNT_FAM_MEMBERS': 2.0},
        'farmer': {'DAYS_BIRTH': -14_600, 'DAYS_EMPLOYED': -5_000, 'CNT_FAM_MEMBERS': 5.0},
    }
    pd_vals = profile_defaults.get(profile_type, profile_defaults['gig'])

    days_birth    = pd_vals['DAYS_BIRTH']
    days_employed = pd_vals['DAYS_EMPLOYED']
    cnt_fam       = pd_vals['CNT_FAM_MEMBERS']

    # Engineered features (same formulas as 03_features.ipynb)
    credit_income_ratio   = loan_amount_inr / (avg_income * 12)
    annuity_income_ratio  = annuity / avg_income
    credit_term           = loan_amount_inr / annuity if annuity > 0 else np.nan
    days_employed_ratio   = days_employed / days_birth if days_birth != 0 else np.nan
    ext_source_mean       = np.nan    # no bureau data — thin-file borrower
    ext_source_min        = np.nan
    income_per_person     = (avg_income * 12) / cnt_fam if cnt_fam > 0 else np.nan
    flag_missing_ext      = 1         # all EXT_SOURCE missing by definition

    row = {
        'AMT_CREDIT':            loan_amount_inr,
        'AMT_ANNUITY':           annuity,
        'AMT_INCOME_TOTAL':      avg_income * 12,
        'DAYS_BIRTH':            days_birth,
        'DAYS_EMPLOYED':         days_employed,
        'CNT_FAM_MEMBERS':       cnt_fam,
        # Engineered
        'CREDIT_INCOME_RATIO':   credit_income_ratio,
        'ANNUITY_INCOME_RATIO':  annuity_income_ratio,
        'CREDIT_TERM':           credit_term,
        'DAYS_EMPLOYED_RATIO':   days_employed_ratio,
        'EXT_SOURCE_MEAN':       ext_source_mean,
        'EXT_SOURCE_MIN':        ext_source_min,
        'INCOME_PER_PERSON':     income_per_person,
        'FLAG_MISSING_EXT':      flag_missing_ext,
        # Metadata (not fed to model)
        '_profile_type':         profile_type,
        '_avg_monthly_income':   avg_income,
        '_income_volatility':    income_std / avg_income if avg_income > 0 else np.nan,
        '_min_monthly_income':   min_income,
    }
    row.update(_SCHEMA_DEFAULTS)
    return row


# ── Decision rules ────────────────────────────────────────────────────────────

def make_decision(default_prob: float) -> str:
    """
    Convert a default probability to an approve / refer / decline decision.

    Thresholds are illustrative and SYNTHETIC — not calibrated to any real
    lender's credit policy.

    Parameters
    ----------
    default_prob : float
        Predicted probability of default (0–1).

    Returns
    -------
    str  — 'Approve', 'Refer', or 'Decline'.
    """
    if default_prob < 0.25:
        return 'Approve'
    elif default_prob < 0.50:
        return 'Refer'
    else:
        return 'Decline'
