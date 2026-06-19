"""
classify_rf.py — Random Forest classifier, same 5-fold CV setup as the SVM
for a fair comparison.
"""
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

FEAT_DIR = Path("features")
X = np.load(FEAT_DIR / "X.npy")
y = np.load(FEAT_DIR / "y.npy", allow_pickle=True)

majority = max(np.mean(y == "tizita"), np.mean(y == "bati"))
print(f"Majority-class baseline: {majority:.1%}\n")

# Random Forest: 200 trees. (Scaling isn't strictly needed for trees,
# but we keep the pipeline identical to the SVM for a fair comparison.)
rf = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=200, random_state=42)
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
f1  = cross_val_score(rf, X, y, cv=cv, scoring="f1_macro")

print("Random Forest (200 trees), 5-fold cross-validation:")
print(f"  Accuracy per fold: {np.round(acc, 3)}")
print(f"  Mean accuracy:     {acc.mean():.3f}  (+/- {acc.std():.3f})")
print(f"  Mean macro-F1:     {f1.mean():.3f}  (+/- {f1.std():.3f})")