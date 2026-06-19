import librosa
from pathlib import Path

DATA_DIR = Path("data/raw")

for scale_dir in sorted(DATA_DIR.iterdir()):
    if not scale_dir.is_dir():
        continue
    wavs = sorted(scale_dir.glob("*.wav")) + sorted(scale_dir.glob("*.Wav"))
    print(f"\n=== {scale_dir.name.upper()} : {len(wavs)} file(s) ===")
    durations = []
    for w in wavs[:5]:                       # show first 5 as a sample
        y, sr = librosa.load(w, sr=None)     # sr=None = keep original rate
        durations.append(len(y) / sr)
        print(f"  {w.name:20s} | {sr} Hz | {len(y)/sr:6.1f} s")
    if len(wavs) > 5:
        # quick stats across ALL files
        all_dur = [librosa.get_duration(path=str(w)) for w in wavs]
        print(f"  ... {len(wavs)-5} more")
        print(f"  duration: min {min(all_dur):.1f}s | max {max(all_dur):.1f}s | avg {sum(all_dur)/len(all_dur):.1f}s")