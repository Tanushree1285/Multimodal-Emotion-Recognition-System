# inference/utils.py
import torch
import numpy as np
import librosa

def load_audio(path, sr=16000, duration=3.0):
    y, _ = librosa.load(path, sr=sr)
    max_len = int(sr*duration)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max(0, max_len - len(y))), mode='constant')
    return y
