"""
features_logmel.py — log-Mel spectrograms for one example per scale,
built on the Phase 1 preprocessing pipeline.
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from preprocessing import preprocess, TARGET_SR

DATA_DIR = Path("data/raw")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

N_MELS = 128          # number of Mel bands
N_FFT = 2048
HOP = 512

scales = ["tizita", "bati"]
fig, axes = plt.subplots(len(scales), 1, figsize=(11, 7), sharex=True)

for ax, scale in zip(axes, scales):
    wavs = sorted((DATA_DIR / scale).glob("*.wav")) + \
           sorted((DATA_DIR / scale).glob("*.Wav"))
    y, sr = preprocess(wavs[0])

    # Mel spectrogram, then convert power to dB (log)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP,
        x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )
    ax.set_title(f"{scale.capitalize()} — log-Mel — {wavs[0].name}")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
out = FIG_DIR / "logmel_tizita_vs_bati.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
print(f"log-Mel shape per clip: {S_db.shape}  (n_mels x frames)")
plt.show()