"""
loader.py — load the EMIR dataset (Tizita vs Bati) at a standard sample rate.
Each file is 30 s; original rates are mixed (16 kHz and 44.1 kHz),
so we force everything to 16 kHz mono and normalise amplitude.
"""
import librosa
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw")
TARGET_SR = 16000          # standardise everyone to 16 kHz (the common rate)

def load_dataset(data_dir=DATA_DIR, target_sr=TARGET_SR):
    signals = []           # list of 1-D numpy arrays (the audio)
    labels  = []           # matching list of scale names
    filenames = []         # keep track of which file each came from

    # each sub-folder name IS the label (tizita, bati)
    for scale_dir in sorted(data_dir.iterdir()):
        if not scale_dir.is_dir():
            continue
        label = scale_dir.name
        wavs = sorted(scale_dir.glob("*.wav")) + sorted(scale_dir.glob("*.Wav"))

        print(f"Loading {len(wavs):3d} files for '{label}'...")
        for w in wavs:
            # sr=target_sr forces resampling; mono=True collapses channels
            y, sr = librosa.load(w, sr=target_sr, mono=True)

            # pre-processing: peak normalise to [-1, 1]
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))

            signals.append(y)
            labels.append(label)
            filenames.append(w.name)

    labels = np.array(labels)
    print(f"\nDone. Loaded {len(signals)} signals.")
    for lab in np.unique(labels):
        print(f"  {lab}: {np.sum(labels == lab)} clips")
    return signals, labels, filenames


if __name__ == "__main__":
    signals, labels, filenames = load_dataset()

    # sanity checks
    print("\n--- sanity check ---")
    print("first signal shape:", signals[0].shape)
    print("sample rate:", TARGET_SR, "Hz")
    print("duration:", len(signals[0]) / TARGET_SR, "s")
    print("amplitude range:", signals[0].min().round(3), "to", signals[0].max().round(3))
    print("label of first clip:", labels[0])