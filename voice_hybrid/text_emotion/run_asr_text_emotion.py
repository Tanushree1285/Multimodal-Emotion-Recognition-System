import os
import torch
import numpy as np
from transformers import AutoTokenizer
from .models import TextEmotionModel
from .asr_to_text import recognize_speech_google

# =============================
# Setup paths
# =============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "text_emotion_model.pth")
print(f"📁 Loading model from: {MODEL_PATH}")

# =============================
# Load tokenizer and model
# =============================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Try loading model safely
try:
    model = TextEmotionModel(num_labels=28)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# =============================
# GoEmotions label set
# =============================
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise"
]

# =============================
# Inference pipeline
# =============================
def analyze_speech_emotion():
    if model is None:
        print("⚠️ Model not loaded, skipping emotion analysis.")
        return None

    print("🎤 Speak now... (recording will stop automatically)")
    text = recognize_speech_google()

    if not text:
        print("⚠️ No speech recognized.")
        return None

    print(f"🗣️ Recognized Text: {text}")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.sigmoid(logits).squeeze().numpy()

    # Handle potential shape mismatches
    probs = np.atleast_1d(probs)
    num_outputs = len(probs)
    num_labels = len(EMOTION_LABELS)

    if num_outputs != num_labels:
        print(f"⚠️ Model outputs {num_outputs} values, but {num_labels} labels are defined.")
        print("⚙️ Adjusting to match the smaller size.")
        min_len = min(num_outputs, num_labels)
        probs = probs[:min_len]

    # Get top-3 predicted emotions
    top_indices = np.argsort(probs)[::-1][:3]

    print("\n🎯 Top Emotions Detected:")
    for i in top_indices:
        label = EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else f"Label_{i}"
        print(f"  • {label} ({probs[i]*100:.2f}%)")

    # Return top emotion & confidence
    top_emotion = EMOTION_LABELS[top_indices[0]] if top_indices[0] < len(EMOTION_LABELS) else "Unknown"
    top_conf = float(probs[top_indices[0]])
    return top_emotion, top_conf


if __name__ == "__main__":
    analyze_speech_emotion()
