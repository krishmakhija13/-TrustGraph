"""Create a stratified 15k-row training sample for Streamlit Cloud deployment."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
feat = ROOT / "data" / "processed" / "features_train.csv"

df = pd.read_csv(feat)
df0 = df[df["TARGET"] == 0].sample(13800, random_state=42)
df1 = df[df["TARGET"] == 1].sample(1200,  random_state=42)
sample = pd.concat([df0, df1]).sample(frac=1, random_state=42).reset_index(drop=True)

out = ROOT / "data" / "processed" / "features_sample.csv"
sample.to_csv(out, index=False)
mb = out.stat().st_size / 1e6
print(f"Sample: {len(sample)} rows, {mb:.1f} MB")
print(f"Default rate: {sample['TARGET'].mean():.3%}")
print(f"Saved: {out}")
