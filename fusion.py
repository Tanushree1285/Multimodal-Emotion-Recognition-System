import cv2
import torch
import numpy as np
import threading
import os
from torchvision import transforms
import torch.nn.functional as F

# === Face Emotion Model ===
from utils.model import SimpleCNN

# === Voice/Text Emotion Analyzer ===
from voice_hybrid.text_emotion.run_asr_text_emotion import analyze_speech_emotion


# =============================
# Setup
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_MODEL_PATH = os.path.join( "models", "face_emotion_model.pth")  # <-- update this if different
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load Face Model
face_model = SimpleCNN(num_classes=len(EMOTIONS)).to(device)
state_dict = torch.load(FACE_MODEL_PATH, map_location=device)
face_model.load_state_dict(state_dict)
face_model.eval()

# Face Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Global variable to sync voice result
voice_result = {"emotion": None, "confidence": 0.0}


# =============================
# Helper Functions
# =============================

def predict_face_emotion(face):
    """Predict emotion from a cropped face image."""
    try:
        img = transform(face).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = face_model(img)
            probs = F.softmax(preds, dim=1)
            top_idx = torch.argmax(probs, dim=1).item()
            return EMOTIONS[top_idx], probs[0][top_idx].item()
    except Exception as e:
        print(f"⚠️ Face prediction error: {e}")
        return "Unknown", 0.0


def run_voice_analysis():
    """Run voice emotion detection asynchronously."""
    global voice_result
    print("🎤 Speak now... (recording will stop automatically)")
    try:
        emotion, confidence = analyze_speech_emotion()
        voice_result["emotion"] = emotion
        voice_result["confidence"] = confidence
        print(f"🎙️ Voice Emotion: {emotion} ({confidence*100:.2f}%)")
    except Exception as e:
        print(f"⚠️ Voice analysis failed: {e}")
        voice_result["emotion"] = None
        voice_result["confidence"] = 0.0


# =============================
# Main Fusion Loop
# =============================

def main():
    print("✅ Multimodal Emotion Fusion starting...")
    print("🎥 Press 'q' to quit the window manually.\n")

    cap = cv2.VideoCapture(0)

    voice_thread = threading.Thread(target=run_voice_analysis)
    voice_thread.start()

    latest_face_emotion = "Neutral"
    latest_face_conf = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            emotion, conf = predict_face_emotion(face_roi)
            latest_face_emotion = emotion
            latest_face_conf = conf

            # Draw face box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion} ({conf*100:.1f}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display voice emotion as overlay
        if voice_result["emotion"]:
            text = f"Voice: {voice_result['emotion']} ({voice_result['confidence']*100:.1f}%)"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Fusion result (face + voice)
        fusion_text = f"Fusion: Face={latest_face_emotion} | Voice={voice_result['emotion'] or '---'}"
        cv2.putText(frame, fusion_text, (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.imshow("🧠 Multimodal Emotion Fusion", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n✅ Program complete. Exiting...")


import csv
from datetime import datetime
import os

# === Save emotion result ===
def save_emotion_to_csv(voice_emotion, confidence, top_emotions):
    os.makedirs("results", exist_ok=True)
    filename = "results/emotion_results.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write header only once
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Dominant_Emotion", "Confidence", "Top_3_Emotions"])
        writer.writerow([now, voice_emotion, confidence, str(top_emotions)])

    print(f"📊 Emotion data saved to {filename}")

# Example call (place this right after your "Program complete" print)
save_emotion_to_csv(
    voice_emotion="fear",
    confidence=77.75,
    top_emotions=[("fear", 77.75), ("nervousness", 7.10), ("disgust", 4.70)]
)



if __name__ == "__main__":
    main()
