"""
preprocessing.py — the full pre-processing pipeline for EMIR audio.
Order: load @16kHz mono  ->  DC offset removal  ->  pre-emphasis
       ->  peak normalisation.  (Windowing is applied per-frame at STFT time.)
"""
import librosa
import numpy as np

TARGET_SR = 16000
PRE_EMPHASIS = 0.97          # standard pre-emphasis coefficient

def preprocess(path, target_sr=TARGET_SR, pre_emph=PRE_EMPHASIS):
    # 1. load at standard rate, mono (handles the mixed 16/44.1 kHz issue)
    y, sr = librosa.load(path, sr=target_sr, mono=True)

    # 2. DC offset removal — centre the signal on zero
    y = y - np.mean(y)

    # 3. pre-emphasis — boost high frequencies: y[n] - alpha * y[n-1]
    y = np.append(y[0], y[1:] - pre_emph * y[:-1])

    # 4. peak normalisation — loudest point -> 1.0
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    return y, sr


if __name__ == "__main__":
    # demo on one file, showing the effect of each step
    from pathlib import Path
    example = sorted((Path("data/raw/tizita")).glob("*.wav"))[0]

    # raw load for comparison
    raw, sr = librosa.load(example, sr=TARGET_SR, mono=True)
    proc, _ = preprocess(example)

    print(f"file: {example.name}")
    print(f"raw  -> mean: {raw.mean():+.6f}  peak: {np.max(np.abs(raw)):.3f}")
    print(f"proc -> mean: {proc.mean():+.6f}  peak: {np.max(np.abs(proc)):.3f}")
    print("\nNote: processed mean is ~0 (DC removed) and peak is 1.0 (normalised).")