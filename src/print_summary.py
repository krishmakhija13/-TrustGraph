import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

ROOT = Path(r'C:\Users\krish\Downloads\TrustGraph')
P    = ROOT / 'data' / 'processed'

# ── 1. Model Comparison ────────────────────────────────────────────────────
print('=' * 68)
print('  1. MODEL COMPARISON ACROSS NOTEBOOKS')
print('=' * 68)
b = pd.read_csv(P / 'baseline_results.csv')
f = pd.read_csv(P / 'feature_results.csv')
g = pd.read_csv(P / 'gbm_results.csv')
print(b[['Model', 'AUC_ROC', 'KS_Statistic']].to_string(index=False))
print(f[['Model', 'AUC_ROC', 'KS_Statistic', 'AUC_Delta', 'KS_Delta']].to_string(index=False))
print(g[['Model', 'AUC_ROC', 'Gini', 'KS_Statistic', 'Avg_Precision', 'Best_Iteration']].to_string(index=False))

# ── 2. Scorecard vs LightGBM ───────────────────────────────────────────────
print()
print('=' * 68)
print('  2. SCORECARD vs LIGHTGBM (Notebook 05)')
print('=' * 68)
lgbm_auc, lgbm_ks = 0.7707, 0.4049

with open(ROOT / 'models' / 'scorecard_final.pkl', 'rb') as fp:
    sc_bundle = pickle.load(fp)
df     = pd.read_csv(P / 'features_train.csv')
feats  = sc_bundle['features']
woe_pl = sc_bundle['woe_pipeline']
lr     = sc_bundle['lr_model']
y      = df['TARGET']
X      = df[feats]
_, Xv, _, yv = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

out = {}
for feat, info in woe_pl.items():
    tbl     = info['table']
    bps     = info['breakpoints']
    vals    = Xv[feat].values.astype(float)
    result  = np.zeros(len(vals))
    miss_r  = tbl[tbl['Bin'] == 'MISSING']
    miss_woe = float(miss_r['WoE'].iloc[0]) if len(miss_r) else 0.0
    non_m   = tbl[tbl['Bin'] != 'MISSING'].reset_index(drop=True)
    woe_arr = non_m['WoE'].values.astype(float)
    nan_m   = np.isnan(vals)
    result[nan_m] = miss_woe
    if len(bps) >= 2 and len(woe_arr) > 0:
        bi = np.digitize(vals[~nan_m], bps[1:], right=False)
        bi = np.clip(bi, 0, len(woe_arr) - 1)
        result[~nan_m] = woe_arr[bi]
    out[f'{feat}_WOE'] = result

Xv_woe = pd.DataFrame(out, index=Xv.index)
sc_prob = lr.predict_proba(Xv_woe)[:, 1]
sc_auc  = roc_auc_score(yv, sc_prob)
sc_ks,_ = ks_2samp(sc_prob[yv == 1], sc_prob[yv == 0])

cmp = pd.DataFrame({
    'Model':          ['LightGBM (04)', 'WOE Scorecard (05)'],
    'AUC_ROC':        [lgbm_auc, round(sc_auc, 4)],
    'KS_Statistic':   [lgbm_ks,  round(sc_ks,  4)],
    'Gini':           [round(2*lgbm_auc-1, 4), round(2*sc_auc-1, 4)],
    'Interpretable':  ['No', 'Yes — points table'],
})
print(cmp.to_string(index=False))
print(f'  AUC gap: {lgbm_auc - sc_auc:+.4f}   KS gap: {lgbm_ks - sc_ks:+.4f}')

# ── 3. India Profiles ─────────────────────────────────────────────────────
print()
print('=' * 68)
print('  3. INDIA PROFILES — SCORES & DECISIONS [ALL DATA SYNTHETIC]')
print('=' * 68)
dec = pd.read_csv(P / 'india_decisions.csv')
print(dec.to_string(index=False))
print()
rc = pd.read_csv(P / 'reason_codes.csv')
for _, row in rc.iterrows():
    prob = row['default_prob']
    dec_label = row['decision']
    btype = row['borrower_type']
    reason = row['reason_code']
    print(f"  [{btype}]  P(default)={prob:.4f}  Decision: {dec_label}")
    print(f"  Reason: {reason}")
    print()

# ── 4. Fairness ────────────────────────────────────────────────────────────
print('=' * 68)
print('  4. FAIRNESS METRICS — GENDER (Notebook 08)')
print('=' * 68)
fair = pd.read_csv(P / 'fairness_results.csv')
print(fair.to_string(index=False))

# ── 5. PSI Stability ──────────────────────────────────────────────────────
print()
print('=' * 68)
print('  5. PSI STABILITY (Notebook 09)')
print('=' * 68)
stab = pd.read_csv(P / 'stability_results.csv')
cols = ['feature', 'PSI', 'status', 'auc_original', 'auc_shifted', 'auc_delta']
print(stab[cols].to_string(index=False))
unstable = stab[stab['status'] == 'UNSTABLE']['feature'].tolist()
monitor  = stab[stab['status'] == 'MONITOR']['feature'].tolist()
print()
print(f"  Overall verdict : {'STABLE' if not unstable else 'UNSTABLE'}")
print(f"  Unstable (PSI>0.20): {unstable if unstable else 'None'}")
print(f"  Monitor  (PSI 0.10-0.20): {monitor if monitor else 'None'}")
print(f"  AUC under shift: {stab['auc_shifted'].iloc[0]:.4f} (delta {stab['auc_delta'].iloc[0]:+.4f})")
print('=' * 68)
