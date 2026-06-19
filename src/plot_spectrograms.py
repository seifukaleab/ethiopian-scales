"""
plot_spectrograms.py — STFT spectrograms using the full preprocessing pipeline,
with the Hann window set explicitly (windowing made visible).
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from preprocessing import preprocess, TARGET_SR   # reuse our pipeline

DATA_DIR = Path("data/raw")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

scales = ["tizita", "bati"]

fig, axes = plt.subplots(len(scales), 1, figsize=(11, 7), sharex=True)

for ax, scale in zip(axes, scales):
    wavs = sorted((DATA_DIR / scale).glob("*.wav")) + \
           sorted((DATA_DIR / scale).glob("*.Wav"))
    example = wavs[0]

    # use our full preprocessing pipeline (DC removal, pre-emphasis, normalise)
    y, sr = preprocess(example)

    # --- STFT with EXPLICIT Hann window (this is the "windowing" step) ---
    D = librosa.stft(y, n_fft=2048, hop_length=512, window="hann")
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=512,
        x_axis="time", y_axis="hz", ax=ax, cmap="magma"
    )
    ax.set_title(f"{scale.capitalize()}  —  {example.name}")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()

out = FIG_DIR / "spectrograms_tizita_vs_bati.png"
plt.savefig(out, dpi=150)
print(f"Saved figure to {out}")
plt.show()