"""
confusion_matrices.py — confusion matrices for SVM and Random Forest,
using cross-validated predictions (every clip predicted when in a test fold).
"""
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

FEAT_DIR = Path("features")
FIG_DIR  = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
X = np.load(FEAT_DIR / "X.npy")
y = np.load(FEAT_DIR / "y.npy", allow_pickle=True)
labels = ["bati", "tizita"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
    "Random Forest": make_pipeline(StandardScaler(),
                                   RandomForestClassifier(n_estimators=200, random_state=42)),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, model) in zip(axes, models.items()):
    # cross_val_predict: each clip predicted when it's in the held-out fold
    y_pred = cross_val_predict(model, X, y, cv=cv)
    cm = confusion_matrix(y, y_pred, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}")
    print(f"\n=== {name} ===")
    print(classification_report(y, y_pred, target_names=labels, digits=3))

plt.tight_layout()
out = FIG_DIR / "confusion_matrices.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()