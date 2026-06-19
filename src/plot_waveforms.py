"""
plot_waveforms.py — visualise one example waveform per scale.
Saves the figure to figures/ for the report.
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

TARGET_SR = 16000
DATA_DIR = Path("data/raw")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# pick the first file from each scale folder as a representative example
scales = ["tizita", "bati"]

fig, axes = plt.subplots(len(scales), 1, figsize=(10, 6), sharex=True)

for ax, scale in zip(axes, scales):
    # grab the first wav in this scale's folder
    wavs = sorted((DATA_DIR / scale).glob("*.wav")) + \
           sorted((DATA_DIR / scale).glob("*.Wav"))
    example = wavs[0]

    y, sr = librosa.load(example, sr=TARGET_SR, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))          # same normalisation as the loader

    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(f"{scale.capitalize()}  —  {example.name}")
    ax.set_ylabel("Amplitude")

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()

out = FIG_DIR / "waveforms_tizita_vs_bati.png"
plt.savefig(out, dpi=150)
print(f"Saved figure to {out}")
plt.show()