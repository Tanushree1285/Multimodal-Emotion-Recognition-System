# main.py
import cv2
import time
from face_module import FaceEmotionDetector
# from voice_module import VoiceEmotionDetector   # To integrate later
# from fusion import MultimodalFusion            # To integrate later
# from report_generator import ReportGenerator   # To generate CSV/PDF reports

def main():
    """
    Entry point for Behavioral Cue Detection system.
    Runs face emotion detection and collects session data.
    """
    # Initialize detectors
    face_detector = FaceEmotionDetector(model_path="models/face_emotion_model.pth")
    # voice_detector = VoiceEmotionDetector()  # Uncomment when voice module is ready

    cap = cv2.VideoCapture(0)
    session_data = []

    # For FPS calculation
    prev_time = time.time()

    print("Behavioral Cue Detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional: resize for faster inference
        frame = cv2.resize(frame, (640, 480))

        # Face Emotion Detection
        faces = face_detector.predict_emotion(frame)

        if faces:
            for face in faces:
                x1, y1, x2, y2 = face['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{face['emotion']}:{face['confidence']:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Save session info
                session_data.append({
                    'modality': 'face',
                    'emotion': face['emotion'],
                    'confidence': face['confidence']
                })

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display
        cv2.imshow("Behavioral Cue Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # TODO: Voice Detection
    # voice_data = voice_detector.record_and_predict()

    # TODO: Multimodal Fusion
    # fused_results = MultimodalFusion(face_data=session_data, voice_data=voice_data).fuse()

    # TODO: Generate Report
    # ReportGenerator(fused_results).save_csv_pdf("reports/session_01")

    print(f"Session ended. {len(session_data)} face emotions detected.")

if __name__ == "__main__":
    main()
