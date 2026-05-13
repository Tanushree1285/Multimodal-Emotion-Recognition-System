import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from utils.model import SimpleCNN
import os
import csv
from datetime import datetime
import time


class FaceEmotionDetector:
    def __init__(self, model_path="models/face_emotion_model.pth", device=None, smooth_window=5):
        """
        Initialize MediaPipe face detector and load the PyTorch emotion recognition model.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.model = SimpleCNN(num_classes=7).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.smooth_window = smooth_window
        self.recent_preds = []

        # CSV logging setup
        os.makedirs("reports", exist_ok=True)
        self.csv_file = "reports/session_01.csv"
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'bbox', 'emotion', 'confidence'])
                writer.writeheader()

        self.log_buffer = []  # Buffer for averaging 5 frames

    def preprocess_face(self, face_img):
        """
        Convert face to grayscale, apply histogram equalization, resize to 48x48, normalize, and convert to tensor.
        """
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.equalizeHist(face_img)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float32') / 255.0
        face_tensor = torch.tensor(face_img).unsqueeze(0).unsqueeze(0)  # 1x1x48x48
        return face_tensor.to(self.device)

    def predict_emotion(self, frame):
        """
        Detect faces in a frame and predict emotions.
        Returns list of dicts with bounding box, emotion, confidence.
        Also logs results to CSV with timestamp (averaged over 5 frames).
        """
        results_list = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.mp_face.process(rgb_frame)

        if detections.detections:
            for detection in detections.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x1 = int(max(bboxC.xmin * iw, 0))
                y1 = int(max(bboxC.ymin * ih, 0))
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                x2, y2 = x1 + w, y1 + h

                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                face_tensor = self.preprocess_face(face_img)
                with torch.no_grad():
                    outputs = self.model(face_tensor)
                    probs = F.softmax(outputs, dim=1)
                    conf, idx = torch.max(probs, 1)
                    emotion = self.emotion_labels[idx.item()]
                    confidence = conf.item()

                # Add to smoothing buffer
                self.recent_preds.append((emotion, confidence))
                if len(self.recent_preds) > self.smooth_window:
                    self.recent_preds.pop(0)

                # Compute smoothed emotion
                emotions, confidences = zip(*self.recent_preds)
                unique, counts = np.unique(emotions, return_counts=True)
                smoothed_emotion = unique[np.argmax(counts)]
                smoothed_conf = np.mean([c for e, c in self.recent_preds if e == smoothed_emotion])

                results_list.append({
                    'bbox': (x1, y1, x2, y2),
                    'emotion': smoothed_emotion,
                    'confidence': smoothed_conf
                })

        # --- Logging to CSV averaged over 5 frames ---
        self.log_buffer.append(results_list)

        if len(self.log_buffer) >= 5:
            avg_results = []
            for i in range(len(results_list)):
                bboxes = [frame[i]['bbox'] for frame in self.log_buffer if i < len(frame)]
                emotions = [frame[i]['emotion'] for frame in self.log_buffer if i < len(frame)]
                confidences = [frame[i]['confidence'] for frame in self.log_buffer if i < len(frame)]

                if not bboxes:
                    continue

                avg_bbox = bboxes[-1]
                unique, counts = np.unique(emotions, return_counts=True)
                avg_emotion = unique[np.argmax(counts)]
                avg_conf = np.mean([c for e, c in zip(emotions, confidences) if e == avg_emotion])

                avg_results.append({
                    'bbox': avg_bbox,
                    'emotion': avg_emotion,
                    'confidence': avg_conf
                })

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'bbox', 'emotion', 'confidence'])
                for face in avg_results:
                    writer.writerow({
                        'timestamp': timestamp,
                        'bbox': face['bbox'],
                        'emotion': face['emotion'],
                        'confidence': round(face['confidence'], 4)
                    })

            self.log_buffer = []

        return results_list


# -------------------------------
# Main Test Runner
# -------------------------------
if __name__ == "__main__":
    detector = FaceEmotionDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        exit()

    print("✅ Face Emotion Detector running... Press 'q' to quit.")
    prev_time = time.time()

    # Emotion color map for visualization
    color_map = {
        'Angry': (0, 0, 255),
        'Disgust': (0, 128, 0),
        'Fear': (128, 0, 128),
        'Happy': (0, 255, 255),
        'Sad': (255, 0, 0),
        'Surprise': (255, 255, 0),
        'Neutral': (200, 200, 200)
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.predict_emotion(frame)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Draw results
        for res in results:
            (x1, y1, x2, y2) = res['bbox']
            emotion = res['emotion']
            conf = res['confidence']
            color = color_map.get(emotion, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{emotion} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Face Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🟢 Session ended. Logged emotions to reports/session_01.csv")
