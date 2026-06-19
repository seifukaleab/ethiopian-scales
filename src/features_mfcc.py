"""
features_mfcc.py — MFCCs with cepstral mean normalisation (CMN),
visualised for one example per scale. Built on Phase 1 preprocessing.
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from preprocessing import preprocess

DATA_DIR = Path("data/raw")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

N_MFCC = 20
N_FFT  = 2048
HOP    = 512

def mfcc_with_cmn(y, sr):
    # 1. compute MFCCs (the "zipped" Mel spectrogram)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                n_fft=N_FFT, hop_length=HOP)
    # 2. cepstral mean normalisation: subtract the per-coefficient mean
    mfcc_cmn = mfcc - np.mean(mfcc, axis=1, keepdims=True)
    return mfcc, mfcc_cmn

scales = ["tizita", "bati"]
fig, axes = plt.subplots(len(scales), 2, figsize=(13, 7))

for row, scale in enumerate(scales):
    wavs = sorted((DATA_DIR / scale).glob("*.wav")) + \
           sorted((DATA_DIR / scale).glob("*.Wav"))
    y, sr = preprocess(wavs[0])
    mfcc, mfcc_cmn = mfcc_with_cmn(y, sr)

    for col, (data, label) in enumerate([(mfcc, "MFCC"), (mfcc_cmn, "MFCC + CMN")]):
        ax = axes[row, col]
        img = librosa.display.specshow(data, sr=sr, hop_length=HOP,
                                       x_axis="time", ax=ax, cmap="coolwarm")
        ax.set_title(f"{scale.capitalize()} — {label}")
        ax.set_ylabel("MFCC coeff")
        fig.colorbar(img, ax=ax)

plt.tight_layout()
out = FIG_DIR / "mfcc_tizita_vs_bati.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
print(f"MFCC shape per clip: {mfcc.shape}  (n_mfcc x frames)")
plt.show()