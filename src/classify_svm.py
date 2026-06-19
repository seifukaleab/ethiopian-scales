"""
classify_svm.py — SVM classifier with proper 5-fold cross-validation.
Scaling is done INSIDE each fold (via Pipeline) to avoid data leakage.
"""
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

FEAT_DIR = Path("features")
X = np.load(FEAT_DIR / "X.npy")
y = np.load(FEAT_DIR / "y.npy", allow_pickle=True)
print(f"Data: {X.shape}, classes: {dict(zip(*np.unique(y, return_counts=True)))}")

# baseline to beat: always guessing the majority class
majority = max(np.mean(y == "tizita"), np.mean(y == "bati"))
print(f"Majority-class baseline: {majority:.1%}\n")

# pipeline: scale inside each fold, then SVM with RBF kernel
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))

# stratified 5-fold keeps class balance in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc = cross_val_score(svm, X, y, cv=cv, scoring="accuracy")
f1  = cross_val_score(svm, X, y, cv=cv, scoring="f1_macro")

print("SVM (RBF kernel), 5-fold cross-validation:")
print(f"  Accuracy per fold: {np.round(acc, 3)}")
print(f"  Mean accuracy:     {acc.mean():.3f}  (+/- {acc.std():.3f})")
print(f"  Mean macro-F1:     {f1.mean():.3f}  (+/- {f1.std():.3f})")