"""
cluster_chroma_only.py — repeat k-means using ONLY the chroma features,
to test whether the scale signal is cleaner without the MFCC/timbre
(confound-carrying) features.
"""
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

FEAT_DIR = Path("features")
X = np.load(FEAT_DIR / "X.npy")
y = np.load(FEAT_DIR / "y.npy", allow_pickle=True)

# column layout (from build_features.py):
#   0..19   mfcc means
#   20..39  mfcc stds
#   40..51  chroma means
#   52..63  chroma stds
chroma_cols = list(range(40, 64))      # the 24 chroma features
mfcc_cols   = list(range(0, 40))       # the 40 mfcc features

def cluster_and_score(X_subset, name):
    Xs = StandardScaler().fit_transform(X_subset)
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = km.fit_predict(Xs)
    ari = adjusted_rand_score(y, clusters)
    print(f"\n{name}")
    print(f"  features used: {X_subset.shape[1]}")
    print(f"  ARI: {ari:.3f}")
    for c in [0, 1]:
        row = {lab: int(np.sum((clusters==c) & (y==lab))) for lab in np.unique(y)}
        print(f"  cluster {c}: {row}")
    return ari

print("Comparing what each feature group clusters on:")
a_all    = cluster_and_score(X,                  "ALL 64 features")
a_chroma = cluster_and_score(X[:, chroma_cols],  "CHROMA only (24)")
a_mfcc   = cluster_and_score(X[:, mfcc_cols],    "MFCC only (40)")

print("\n--- summary ---")
print(f"  all:    ARI {a_all:.3f}")
print(f"  chroma: ARI {a_chroma:.3f}")
print(f"  mfcc:   ARI {a_mfcc:.3f}")