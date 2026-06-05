"""
explainability.py — SHAP-based reason code generation for TrustGraph.

Converts raw SHAP values into plain-English sentences readable by loan officers,
without requiring any knowledge of machine learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# Human-readable descriptions for each model feature
FEATURE_DESCRIPTIONS: Dict[str, Tuple[str, str]] = {
    # (positive_label, negative_label)
    # positive = feature pushes toward approval (lower default risk)
    # negative = feature pushes toward decline (higher default risk)
    'EXT_SOURCE_MEAN':        ('strong external credit scores',      'weak external credit scores'),
    'EXT_SOURCE_MIN':         ('consistent bureau ratings',          'at least one very low bureau score'),
    'EXT_SOURCE_1':           ('strong primary bureau score',        'weak primary bureau score'),
    'EXT_SOURCE_2':           ('strong secondary bureau score',      'weak secondary bureau score'),
    'EXT_SOURCE_3':           ('strong tertiary bureau score',       'weak tertiary bureau score'),
    'CREDIT_INCOME_RATIO':    ('manageable loan-to-income ratio',    'loan amount high relative to income'),
    'ANNUITY_INCOME_RATIO':   ('affordable monthly repayments',      'debt obligations exceed safe income share'),
    'CREDIT_TERM':            ('standard loan term length',          'very long repayment horizon'),
    'DAYS_EMPLOYED_RATIO':    ('strong employment stability',        'limited employment history'),
    'DAYS_EMPLOYED':          ('long tenure at current employer',    'short or no employment tenure'),
    'DAYS_BIRTH':             ('experienced borrower age profile',   'very young or very old applicant'),
    'AMT_CREDIT':             ('loan amount within normal range',    'unusually high loan amount requested'),
    'AMT_INCOME_TOTAL':       ('solid annual income',                'low annual income'),
    'AMT_ANNUITY':            ('affordable annuity payment',         'high annuity relative to profile'),
    'INCOME_PER_PERSON':      ('good income per household member',   'income stretched across many dependants'),
    'FLAG_MISSING_EXT':       ('bureau data available',              'missing credit bureau data (thin file)'),
    'FLAG_OWN_REALTY':        ('owns property (collateral signal)',  'no property ownership'),
    'REGION_RATING_CLIENT':   ('favourable region risk rating',      'high-risk region rating'),
    'OWN_CAR_AGE':            ('vehicle ownership history',          'no vehicle or very old vehicle'),
    'DAYS_ID_PUBLISH':        ('recently updated identity document', 'outdated identity document'),
}

_DEFAULT_DESC = ('positive applicant profile',  'elevated risk indicator')


def reason_code_generator(
    shap_values: np.ndarray,
    feature_names: List[str],
    default_prob: float,
    top_n: int = 3,
) -> str:
    """
    Generate 2–3 plain-English sentences explaining a single credit decision.

    Takes the SHAP values for one borrower and returns a reason-code string
    that a loan officer can read and act on without ML expertise.

    Parameters
    ----------
    shap_values : np.ndarray, shape (n_features,)
        SHAP values for a single borrower (positive = increases default probability).
    feature_names : list of str
        Feature names corresponding to shap_values.
    default_prob : float
        Model's predicted default probability for this borrower (0–1).
    top_n : int
        Number of driving factors to surface (default 3).

    Returns
    -------
    str
        Plain-English decision explanation.
    """
    pairs = list(zip(feature_names, shap_values))

    # Top factors increasing risk (positive SHAP = pushes toward default)
    risk_drivers = sorted(
        [(name, sv) for name, sv in pairs if sv > 0],
        key=lambda x: -x[1]
    )[:top_n]

    # Top factors reducing risk (negative SHAP = pushes toward approval)
    protection_drivers = sorted(
        [(name, sv) for name, sv in pairs if sv < 0],
        key=lambda x: x[1]
    )[:top_n]

    def _label(name: str, positive: bool) -> str:
        desc = FEATURE_DESCRIPTIONS.get(name, _DEFAULT_DESC)
        return desc[0] if positive else desc[1]

    if default_prob < 0.25:
        verdict = "Approved"
        strengths = [_label(n, True) for n, _ in protection_drivers[:2]]
        weaknesses = [_label(n, False) for n, _ in risk_drivers[:1]] if risk_drivers else []
        strength_str = ', '.join(strengths) if strengths else 'overall strong profile'
        sentence1 = f"{verdict} — {strength_str}."
        sentence2 = (
            f"Minor risk flags noted: {weaknesses[0]}."
            if weaknesses else
            "No significant risk flags identified."
        )
        sentence3 = f"Predicted default probability: {default_prob:.1%}."

    elif default_prob < 0.50:
        verdict = "Refer for manual review"
        strength_str = _label(protection_drivers[0][0], True) if protection_drivers else 'some positive signals'
        risk_str = ', '.join([_label(n, False) for n, _ in risk_drivers[:2]]) if risk_drivers else 'mixed risk profile'
        sentence1 = f"{verdict} — borderline risk profile."
        sentence2 = f"Strengths: {strength_str}. Concerns: {risk_str}."
        sentence3 = f"Predicted default probability: {default_prob:.1%} — recommend additional income verification."

    else:
        verdict = "Declined"
        risk_str = ', '.join([_label(n, False) for n, _ in risk_drivers[:2]]) if risk_drivers else 'high overall risk'
        mitigant = _label(protection_drivers[0][0], True) if protection_drivers else 'limited mitigating factors'
        sentence1 = f"{verdict} — {risk_str}."
        sentence2 = f"Mitigating factor: {mitigant}, but insufficient to offset risk profile."
        sentence3 = f"Predicted default probability: {default_prob:.1%}."

    return ' '.join([sentence1, sentence2, sentence3])


def batch_reason_codes(
    shap_matrix: np.ndarray,
    feature_names: List[str],
    default_probs: np.ndarray,
    borrower_ids: Optional[List] = None,
) -> pd.DataFrame:
    """
    Run reason_code_generator over a batch of borrowers.

    Parameters
    ----------
    shap_matrix : np.ndarray, shape (n_borrowers, n_features)
    feature_names : list of str
    default_probs : np.ndarray, shape (n_borrowers,)
    borrower_ids : list, optional

    Returns
    -------
    pd.DataFrame with columns: borrower_id, default_prob, decision, reason_code.
    """
    from india_layer import make_decision   # local import to avoid circular dep

    n = shap_matrix.shape[0]
    ids = borrower_ids if borrower_ids is not None else list(range(n))

    records = []
    for i, (bid, prob) in enumerate(zip(ids, default_probs)):
        code = reason_code_generator(shap_matrix[i], feature_names, prob)
        records.append({
            'borrower_id':   bid,
            'default_prob':  round(float(prob), 4),
            'decision':      make_decision(prob),
            'reason_code':   code,
        })
    return pd.DataFrame(records)
