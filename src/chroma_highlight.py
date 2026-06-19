"""
chroma_highlight.py — VISUALISATION ONLY.
Keeps the most prominent pitch classes and dims the rest, to make the
dominant notes of each scale easier to see. NOT used for modelling —
the classifier uses the full chroma.
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

HOP = 512
KEEP_TOP = 5          # pentatonic ~ 5 notes; keep the 5 strongest pitch classes
PITCHES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

scales = ["tizita", "bati"]
fig, axes = plt.subplots(len(scales), 1, figsize=(11, 7), sharex=True)

for ax, scale in zip(axes, scales):
    wavs = sorted((DATA_DIR / scale).glob("*.wav")) + \
           sorted((DATA_DIR / scale).glob("*.Wav"))
    y, sr = preprocess(wavs[0])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP)

    # --- find the dominant pitch classes over the WHOLE clip ---
    mean_per_pitch = chroma.mean(axis=1)            # average energy per pitch class
    top_idx = np.argsort(mean_per_pitch)[-KEEP_TOP:]  # indices of the strongest 5

    # build a masked copy: keep top rows, zero the rest
    chroma_masked = np.zeros_like(chroma)
    chroma_masked[top_idx, :] = chroma[top_idx, :]

    img = librosa.display.specshow(chroma_masked, sr=sr, hop_length=HOP,
                                   x_axis="time", y_axis="chroma",
                                   ax=ax, cmap="magma")
    kept = sorted([PITCHES[i] for i in top_idx])
    ax.set_title(f"{scale.capitalize()} — top {KEEP_TOP} pitch classes: {', '.join(kept)}")
    fig.colorbar(img, ax=ax)

plt.tight_layout()
out = FIG_DIR / "chroma_highlight_tizita_vs_bati.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()