"""
explain_lime.py — Phase 5, step 3.
LIME explains a SINGLE clip's prediction: which features pushed THIS
clip toward Tizita or Bati. Local, per-instance explanation.
"""
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

FEAT_DIR = Path("features")
FIG_DIR  = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
X = np.load(FEAT_DIR / "X.npy")
y = np.load(FEAT_DIR / "y.npy", allow_pickle=True)

names  = [f"mfcc{i}_mean" for i in range(20)] + [f"mfcc{i}_std" for i in range(20)]
names += [f"chroma{i}_mean" for i in range(12)] + [f"chroma{i}_std" for i in range(12)]

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)
classes = list(rf.classes_)   # e.g. ['bati', 'tizita']

# LIME explainer over the training data
explainer = LimeTabularExplainer(
    X, feature_names=names, class_names=classes,
    discretize_continuous=True, mode="classification"
)

# explain a few example clips (pick one Tizita and one Bati)
tizita_idx = np.where(y == "tizita")[0][0]
bati_idx   = np.where(y == "bati")[0][0]

for idx, true_label in [(tizita_idx, "tizita"), (bati_idx, "bati")]:
    exp = explainer.explain_instance(X[idx], rf.predict_proba, num_features=10)
    pred = classes[np.argmax(rf.predict_proba(X[idx].reshape(1, -1)))]
    print(f"\n=== Clip {idx} | true={true_label} | predicted={pred} ===")
    for feat, weight in exp.as_list():
        direction = classes[1] if weight > 0 else classes[0]
        print(f"  {feat:30s} weight={weight:+.4f}  (-> {direction})")

    # save the figure
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(8, 5)
    fig.tight_layout()
    out = FIG_DIR / f"lime_{true_label}_clip{idx}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")