import cv2
import threading
import time
import tkinter as tk
from tkinter import ttk
from collections import defaultdict
from voice_hybrid.text_emotion.run_asr_text_emotion import analyze_speech_emotion

emotion_stats = defaultdict(list)
latest_emotion = ""
latest_conf = 0.0

# ===========================
# Tkinter UI Setup
# ===========================
root = tk.Tk()
root.title("🎯 Emotion Detection Dashboard")
root.geometry("420x300")
root.configure(bg="#F5F5FF")

title_label = tk.Label(root, text="Multimodal Emotion Detector", font=("Helvetica", 16, "bold"), bg="#F5F5FF", fg="#4B0082")
title_label.pack(pady=15)

emotion_label = tk.Label(root, text="Emotion: --", font=("Helvetica", 14), bg="#F5F5FF", fg="#006400")
emotion_label.pack(pady=10)

confidence_label = tk.Label(root, text="Confidence: --%", font=("Helvetica", 12), bg="#F5F5FF")
confidence_label.pack(pady=5)

progress = ttk.Progressbar(root, length=300, mode="determinate")
progress.pack(pady=10)

status_label = tk.Label(root, text="Status: Waiting for input...", font=("Helvetica", 11), bg="#F5F5FF", fg="#333")
status_label.pack(pady=5)

# ===========================
# Update UI function
# ===========================
def update_ui(emotion, confidence):
    emotion_label.config(text=f"Emotion: {emotion}")
    confidence_label.config(text=f"Confidence: {confidence*100:.2f}%")
    progress["value"] = confidence * 100
    status_label.config(text="Status: Detection Complete ✅")
    root.update()

# ===========================
# Voice detection thread
# ===========================
def run_voice_thread():
    global latest_emotion, latest_conf
    status_label.config(text="🎤 Listening for emotion...")
    root.update()
    result = analyze_speech_emotion()

    if result:
        top_emotion, top_conf = result
        latest_emotion, latest_conf = top_emotion, top_conf
        emotion_stats[top_emotion].append(top_conf)
        update_ui(top_emotion, top_conf)
    else:
        status_label.config(text="⚠️ No speech recognized.")
        root.update()

# ===========================
# Camera + Detection main
# ===========================
def start_detection():
    status_label.config(text="📸 Starting detection...")
    root.update()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_label.config(text="⚠️ Could not open camera.")
        return

    voice_thread = threading.Thread(target=run_voice_thread)
    voice_thread.start()

    start_time = time.time()
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("🎥 Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    voice_thread.join()

    status_label.config(text="✅ Detection finished. Close window to see analytics.")
    root.update()

# ===========================
# Button to start
# ===========================
start_btn = tk.Button(root, text="Start Emotion Detection", command=start_detection, bg="#B19CD9", fg="white",
                      font=("Helvetica", 12, "bold"), relief="flat", width=22)
start_btn.pack(pady=20)

# ===========================
# When closed, print analytics
# ===========================
def on_close():
    if emotion_stats:
        print("\n📊 Session Summary:")
        for emotion, values in emotion_stats.items():
            avg_conf = sum(values) / len(values)
            print(f"  • {emotion}: {len(values)} detections (avg {avg_conf*100:.2f}%)")
    else:
        print("⚠️ No emotions recorded this session.")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
