import librosa
import sklearn
import matplotlib
import numpy as np

print("librosa:", librosa.__version__)
print("sklearn:", sklearn.__version__)
print("numpy:", np.__version__)

# load librosa's built-in example clip to confirm audio loading works
y, sr = librosa.load(librosa.example('trumpet'))
print(f"\nLoaded test audio: {len(y)} samples at {sr} Hz = {len(y)/sr:.1f} seconds")
print("Everything works! ✅")