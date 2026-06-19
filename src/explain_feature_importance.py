"""
explain_feature_importance.py — Phase 5, step 1.
Random Forest's built-in feature importance: which of the 64 features
drive the classification, and is it MFCC (timbre) or chroma (pitch class)?
"""
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

FEAT_DIR = Path("features")
FIG_DIR  = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
X = np.load(FEAT_DIR / "X.npy")
y = np.load(FEAT_DIR / "y.npy", allow_pickle=True)

# feature names (same layout as build_features.py)
names  = [f"mfcc{i}_mean" for i in range(20)] + [f"mfcc{i}_std" for i in range(20)]
names += [f"chroma{i}_mean" for i in range(12)] + [f"chroma{i}_std" for i in range(12)]
names = np.array(names)

# train on all data to read importances
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)
imp = rf.feature_importances_

# --- top 15 individual features ---
order = np.argsort(imp)[::-1]
print("Top 15 most important features:")
for i in order[:15]:
    print(f"  {names[i]:15s} {imp[i]:.4f}")

# --- grouped: how much total importance goes to MFCC vs chroma? ---
mfcc_mask   = np.array([n.startswith("mfcc")   for n in names])
chroma_mask = np.array([n.startswith("chroma") for n in names])
print(f"\nTotal importance — MFCC features:   {imp[mfcc_mask].sum():.3f}")
print(f"Total importance — chroma features: {imp[chroma_mask].sum():.3f}")

# --- plot top 15 ---
plt.figure(figsize=(8,6))
top = order[:15][::-1]
colors = ["#534AB7" if names[i].startswith("mfcc") else "#1D9E75" for i in top]
plt.barh(range(len(top)), imp[top], color=colors)
plt.yticks(range(len(top)), names[top])
plt.xlabel("Feature importance")
plt.title("Random Forest — top 15 features\n(purple = MFCC, green = chroma)")
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_importance.png", dpi=150)
print(f"\nSaved {FIG_DIR/'feature_importance.png'}")
plt.show()