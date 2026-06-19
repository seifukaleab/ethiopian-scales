"""
explain_shap.py — Phase 5, step 2.
SHAP values for the Random Forest: how each feature pushes predictions
toward Tizita vs Bati, globally and per-clip.
"""
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

FEAT_DIR = Path("features")
FIG_DIR  = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
X = np.load(FEAT_DIR / "X.npy")
y = np.load(FEAT_DIR / "y.npy", allow_pickle=True)

names  = [f"mfcc{i}_mean" for i in range(20)] + [f"mfcc{i}_std" for i in range(20)]
names += [f"chroma{i}_mean" for i in range(12)] + [f"chroma{i}_std" for i in range(12)]

# train RF on all data
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# TreeExplainer is fast and exact for tree models
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# shap_values for binary RF: list of 2 arrays (one per class) OR a 3D array.
# We take the values for the "tizita" class for the summary.
classes = list(rf.classes_)
print("Classes:", classes)

# handle both possible shapes returned by shap versions
if isinstance(shap_values, list):
    sv = shap_values[classes.index("tizita")]
else:
    # 3D array (n_samples, n_features, n_classes)
    sv = shap_values[:, :, classes.index("tizita")]

# --- global summary plot (beeswarm) ---
plt.figure()
shap.summary_plot(sv, X, feature_names=names, show=False, max_display=15)
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR/'shap_summary.png'}")

# --- bar version (mean absolute SHAP = global importance) ---
plt.figure()
shap.summary_plot(sv, X, feature_names=names, plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR/'shap_bar.png'}")

print("\nDone. Two figures saved: shap_summary.png (beeswarm) and shap_bar.png.")