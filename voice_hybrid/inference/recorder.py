# inference/recorder.py
import sounddevice as sd
import soundfile as sf
import numpy as np

def record(duration=3, sr=16000, out_file="tmp.wav"):
    print("Recording...")
    frames = sd.rec(int(duration*sr), samplerate=sr, channels=1)
    sd.wait()
    frames = frames.flatten()
    sf.write(out_file, frames, sr)
    print("Saved", out_file)
    return out_file
