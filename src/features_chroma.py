"""
features_chroma.py — chroma features (12 pitch classes) per scale.
The feature most directly tied to scale identity; resistant to the
recording-bandwidth confound because it folds all octaves together.
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

N_FFT = 2048
HOP   = 512

scales = ["tizita", "bati"]
fig, axes = plt.subplots(len(scales), 1, figsize=(11, 7), sharex=True)

for ax, scale in zip(axes, scales):
    wavs = sorted((DATA_DIR / scale).glob("*.wav")) + \
           sorted((DATA_DIR / scale).glob("*.Wav"))
    y, sr = preprocess(wavs[0])

    # chroma: energy folded onto the 12 pitch classes
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)

    img = librosa.display.specshow(chroma, sr=sr, hop_length=HOP,
                                   x_axis="time", y_axis="chroma",
                                   ax=ax, cmap="magma")
    ax.set_title(f"{scale.capitalize()} — Chroma — {wavs[0].name}")
    fig.colorbar(img, ax=ax)

plt.tight_layout()
out = FIG_DIR / "chroma_tizita_vs_bati.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
print(f"Chroma shape per clip: {chroma.shape}  (12 pitch classes x frames)")
plt.show()