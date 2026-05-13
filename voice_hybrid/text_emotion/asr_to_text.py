import torch
from transformers import BertTokenizer
from .models import TextEmotionModel
import speech_recognition as sr

# ==========================
# ✅ Load Trained Model
# ==========================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = TextEmotionModel()
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "text_emotion_model.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

model.eval()

# ==========================
# ✅ Speech-to-Text Function
# ==========================
def recognize_speech_google():
    """Record speech using microphone and return transcribed text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Speak something... (listening)")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"🗣️ Recognized Speech: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand the audio.")
        return None
    except sr.RequestError:
        print("⚠️ Error connecting to Google Speech API.")
        return None

# ==========================
# ✅ Emotion Prediction Function
# ==========================
def predict_emotion(text):
    """Classify emotion for given text using trained model."""
    if not text:
        return None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs)
    predicted_label = torch.argmax(logits, dim=1).item()
    emotions = ["happy", "sad", "angry", "neutral"]  # update these based on your dataset
    predicted_emotion = emotions[predicted_label]
    print(f"🧠 Predicted Emotion: {predicted_emotion}")
    return predicted_emotion
