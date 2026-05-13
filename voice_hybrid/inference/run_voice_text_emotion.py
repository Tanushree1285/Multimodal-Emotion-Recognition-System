import torch
from transformers import BertTokenizer
import sounddevice as sd
import scipy.io.wavfile as wav
import os

# Import your ASR module
from inference.asr_to_text import transcribe_audio
from models import TextEmotionModel


# ========== CONFIG ==========
SAMPLE_RATE = 16000
DURATION = 5  # seconds
MODEL_PATH = "models/text_emotion_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== LOAD TEXT MODEL ==========
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TextEmotionModel(num_labels=28)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("🎙️ Speak now... recording for", DURATION, "seconds")

# ========== RECORD AUDIO ==========
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()

wav.write("temp.wav", SAMPLE_RATE, audio)

# ========== SPEECH → TEXT ==========
print("🧠 Transcribing speech...")
text = transcribe_audio("temp.wav")
print(f"🗣️ You said: {text}")

# ========== TEXT → EMOTION ==========
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

with torch.no_grad():
    logits = model(**inputs)
    probs = torch.sigmoid(logits)
    predicted = torch.argmax(probs, dim=1).item()

print(f"❤️ Predicted Emotion Class ID: {predicted}")
