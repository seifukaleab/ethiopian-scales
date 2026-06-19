"""
visualize_pca.py — project the 64-D feature space down to 2-D with PCA,
colour points by true scale. Shows visually why clustering struggled.
"""
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

FEAT_DIR = Path("features")
FIG_DIR  = Path("figures"); FIG_DIR.mkdir(exist_ok=True)

X = np.load(FEAT_DIR / "X.npy")
y = np.load(FEAT_DIR / "y.npy", allow_pickle=True)

X_scaled = StandardScaler().fit_transform(X)

# PCA to 2 dimensions
pca = PCA(n_components=2)
X2 = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_
print(f"Variance explained: PC1 {var[0]:.1%}, PC2 {var[1]:.1%}, total {var.sum():.1%}")

# plot, coloured by true label
plt.figure(figsize=(8,6))
for lab, color in [("tizita", "#534AB7"), ("bati", "#D85A30")]:
    mask = (y == lab)
    plt.scatter(X2[mask,0], X2[mask,1], c=color, label=lab,
                alpha=0.6, edgecolors="white", linewidths=0.3, s=40)
plt.xlabel(f"PC1 ({var[0]:.1%} variance)")
plt.ylabel(f"PC2 ({var[1]:.1%} variance)")
plt.title("Feature space (PCA) — coloured by true scale")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "pca_scatter.png", dpi=150)
print(f"Saved {FIG_DIR/'pca_scatter.png'}")
plt.show()