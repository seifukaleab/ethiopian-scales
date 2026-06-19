"""
build_features.py — extract a fixed-length feature vector per clip across
all 306 clips, and save the feature matrix + labels for Phase 3 onward.

Per clip: MFCC(20) mean+std  +  chroma(12) mean+std  = 64 features.
"""
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from preprocessing import preprocess

DATA_DIR = Path("data/raw")
FEAT_DIR = Path("features")
FEAT_DIR.mkdir(exist_ok=True)

N_MFCC = 20
N_FFT  = 2048
HOP    = 512

def extract_features(y, sr):
    # MFCC + CMN
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)   # cepstral mean normalisation
    # chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)

    # summarise each coefficient over time: mean + std
    feats = np.concatenate([
        mfcc.mean(axis=1),   mfcc.std(axis=1),     # 20 + 20
        chroma.mean(axis=1), chroma.std(axis=1),   # 12 + 12
    ])
    return feats   # length 64

# build readable column names
cols  = [f"mfcc{i}_mean" for i in range(N_MFCC)] + [f"mfcc{i}_std" for i in range(N_MFCC)]
cols += [f"chroma{i}_mean" for i in range(12)]   + [f"chroma{i}_std" for i in range(12)]

X, y_labels, names = [], [], []

for scale_dir in sorted(DATA_DIR.iterdir()):
    if not scale_dir.is_dir():
        continue
    label = scale_dir.name
    wavs = sorted(scale_dir.glob("*.wav")) + sorted(scale_dir.glob("*.Wav"))
    print(f"Extracting {len(wavs):3d} clips for '{label}'...")
    for w in wavs:
        y, sr = preprocess(w)
        X.append(extract_features(y, sr))
        y_labels.append(label)
        names.append(w.name)

X = np.array(X)
y_labels = np.array(y_labels)
print(f"\nFeature matrix: {X.shape}   (clips x features)")
print(f"Labels: {y_labels.shape}  ->  ", {l: int((y_labels==l).sum()) for l in np.unique(y_labels)})

# save: .npy for fast loading, .csv for inspection
np.save(FEAT_DIR / "X.npy", X)
np.save(FEAT_DIR / "y.npy", y_labels)

df = pd.DataFrame(X, columns=cols)
df.insert(0, "label", y_labels)
df.insert(0, "filename", names)
df.to_csv(FEAT_DIR / "features.csv", index=False)

print(f"\nSaved:")
print(f"  features/X.npy        {X.shape}")
print(f"  features/y.npy        {y_labels.shape}")
print(f"  features/features.csv (human-readable)")