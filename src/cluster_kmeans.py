"""
cluster_kmeans.py — unsupervised k-means on the 306x64 feature matrix.
Asks: do the clips naturally fall into groups, and do those groups
match the true Tizita/Bati labels?
"""
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import matplotlib.pyplot as plt

FEAT_DIR = Path("features")
FIG_DIR  = Path("figures"); FIG_DIR.mkdir(exist_ok=True)

# load the feature matrix + labels
X = np.load(FEAT_DIR / "X.npy")
y = np.load(FEAT_DIR / "y.npy", allow_pickle=True)
print(f"Loaded X: {X.shape}, y: {y.shape}")

# IMPORTANT: standardise features first (different features have different scales)
X_scaled = StandardScaler().fit_transform(X)

# --- elbow check: try k = 1..6, plot inertia ---
inertias = []
Ks = range(1, 7)
for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7,4))
plt.plot(list(Ks), inertias, "o-")
plt.xlabel("k (number of clusters)")
plt.ylabel("Inertia (within-cluster spread)")
plt.title("Elbow method")
plt.tight_layout()
plt.savefig(FIG_DIR / "elbow.png", dpi=150)
print(f"Saved {FIG_DIR/'elbow.png'}")

# --- cluster with k=2 (our two scales) ---
km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = km2.fit_predict(X_scaled)

# how well do the 2 clusters match the true labels?
ari = adjusted_rand_score(y, clusters)
print(f"\nAdjusted Rand Index (cluster vs true label): {ari:.3f}")
print("(0 = random, 1 = perfect match to true labels)")

# crosstab: which cluster did each true scale land in?
print("\nCluster vs true label (counts):")
labels_unique = np.unique(y)
for c in [0, 1]:
    row = {lab: int(np.sum((clusters==c) & (y==lab))) for lab in labels_unique}
    print(f"  cluster {c}: {row}")