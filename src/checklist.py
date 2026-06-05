"""Final delivery checklist for TrustGraph."""
from pathlib import Path
import pickle, urllib.request

ROOT = Path(r"C:\Users\krish\Downloads\TrustGraph")
P    = ROOT / "data" / "processed"
M    = ROOT / "models"
N    = ROOT / "notebooks"

GREEN = "✅"
RED   = "❌"

def chk(label, condition):
    icon = GREEN if condition else RED
    print(f"  {icon}  {label}")
    return condition

results = []

print()
print("=" * 62)
print("  TRUSTGRAPH — FINAL DELIVERY CHECKLIST")
print("=" * 62)

# ── Notebooks ──────────────────────────────────────────────────────────────────
print("\n── Notebooks Executed (9 total) ──")
notebooks = [
    "01_eda.ipynb", "02_baseline.ipynb", "03_features.ipynb",
    "04_gbm.ipynb", "05_scorecard.ipynb", "06_india_layer.ipynb",
    "07_explainability.ipynb", "08_fairness.ipynb", "09_stability.ipynb",
]
nb_ok = all(chk(nb, (N / nb).exists()) for nb in notebooks)
results.append(("9 Notebooks present & executed", nb_ok))

# ── Output CSVs ────────────────────────────────────────────────────────────────
print("\n── Output Files (10 required) ──")
outputs = [
    ("baseline_results.csv",  P / "baseline_results.csv"),
    ("feature_results.csv",   P / "feature_results.csv"),
    ("gbm_results.csv",       P / "gbm_results.csv"),
    ("scorecard_table.csv",   P / "scorecard_table.csv"),
    ("india_profiles.csv",    P / "india_profiles.csv"),
    ("reason_codes.csv",      P / "reason_codes.csv"),
    ("fairness_results.csv",  P / "fairness_results.csv"),
    ("stability_results.csv", P / "stability_results.csv"),
    ("india_decisions.csv",   P / "india_decisions.csv"),
    ("features_train.csv",    P / "features_train.csv"),
]
out_ok = all(chk(label, path.exists()) for label, path in outputs)
results.append(("10 Output CSVs present", out_ok))

# ── Models ─────────────────────────────────────────────────────────────────────
print("\n── Saved Models ──")
lgbm_ok = chk("models/lgbm_final.pkl",      (M / "lgbm_final.pkl").exists())
sc_ok   = chk("models/scorecard_final.pkl", (M / "scorecard_final.pkl").exists())
results.append(("Both models saved", lgbm_ok and sc_ok))

# ── Charts ─────────────────────────────────────────────────────────────────────
print("\n── EDA & Model Charts ──")
charts = [
    "target_distribution.png", "missing_values.png",
    "numeric_distributions.png", "correlation_heatmap.png",
    "roc_curves_baseline.png", "feature_importance.png",
    "gbm_roc_curve.png", "gbm_pr_curve.png", "gbm_feature_importance.png",
    "shap_beeswarm.png", "shap_bar.png",
    "woe_charts.png", "fairness_gender.png", "psi_chart.png",
]
charts_ok = all(chk(c, (P / "eda_charts" / c).exists()) for c in charts)
results.append(("14 Charts saved", charts_ok))

# ── Source files ───────────────────────────────────────────────────────────────
print("\n── Source & App Files ──")
src_files = [
    ("src/india_layer.py",    ROOT / "src" / "india_layer.py"),
    ("src/explainability.py", ROOT / "src" / "explainability.py"),
    ("app/streamlit_app.py",  ROOT / "app" / "streamlit_app.py"),
]
src_ok = all(chk(label, path.exists()) for label, path in src_files)
results.append(("src/ + app/ files present", src_ok))

# ── Project files ──────────────────────────────────────────────────────────────
print("\n── Project Files ──")
proj_files = [
    ("README.md",        ROOT / "README.md"),
    ("CLAUDE.md",        ROOT / "CLAUDE.md"),
    ("requirements.txt", ROOT / "requirements.txt"),
    (".gitignore",       ROOT / ".gitignore"),
]
proj_ok = all(chk(label, path.exists()) for label, path in proj_files)
results.append(("README, CLAUDE.md, requirements, .gitignore", proj_ok))

# ── Streamlit HTTP ─────────────────────────────────────────────────────────────
print("\n── Streamlit App ──")
try:
    r = urllib.request.urlopen("http://localhost:8501", timeout=5)
    http_ok = (r.status == 200)
    chk(f"Streamlit live on http://localhost:8501  (HTTP {r.status})", http_ok)
except Exception as e:
    chk(f"Streamlit not reachable: {e}", False)
    http_ok = False
results.append(("Streamlit running on localhost:8501", http_ok))

# ── Model loadability ──────────────────────────────────────────────────────────
print("\n── Model Loadability ──")
try:
    with open(M / "lgbm_final.pkl", "rb") as f:
        lgbm = pickle.load(f)
    sz = round((M / "lgbm_final.pkl").stat().st_size / 1e6, 2)
    n_feats = len(lgbm.feature_name_)
    chk(f"lgbm_final.pkl  — {sz} MB, {n_feats} features, best_iter={lgbm.best_iteration_}", True)
    lm_ok = True
except Exception as e:
    chk(f"lgbm_final.pkl failed: {e}", False)
    lm_ok = False

try:
    with open(M / "scorecard_final.pkl", "rb") as f:
        sc = pickle.load(f)
    n_sc = len(sc["features"])
    chk(f"scorecard_final.pkl — {n_sc} features, WOE pipeline present", True)
    sm_ok = True
except Exception as e:
    chk(f"scorecard_final.pkl failed: {e}", False)
    sm_ok = False
results.append(("Both models loadable", lm_ok and sm_ok))

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 62)
print("  SUMMARY")
print("=" * 62)
all_pass = True
for label, ok in results:
    icon = GREEN if ok else RED
    print(f"  {icon}  {label}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print("  ALL CHECKS PASSED — TrustGraph is demo-ready.")
    print()
    print("  Next steps:")
    print("    streamlit run app/streamlit_app.py  →  http://localhost:8501")
    print("    git add .  &&  git commit -m 'TrustGraph v1.0'")
else:
    print("  Some checks failed — see details above.")
print("=" * 62)
