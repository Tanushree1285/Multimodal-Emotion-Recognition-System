import pandas as pd
import matplotlib.pyplot as plt

# Load session logs with column names
csv_file = "session_01.csv"
df = pd.read_csv(csv_file, header=None, names=["timestamp", "bbox", "emotion", "confidence"])

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- 1. Bar chart of total counts per emotion ---
emotion_counts = df['emotion'].value_counts()
plt.figure(figsize=(8,5))
plt.bar(emotion_counts.index, emotion_counts.values, color='skyblue')
plt.title("Emotion Counts in Session")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.show()

# --- 2. Emotion timeline (confidence over time) ---
plt.figure(figsize=(12,5))
for emotion in df['emotion'].unique():
    subset = df[df['emotion'] == emotion]
    plt.scatter(subset['timestamp'], subset['confidence'], label=emotion, alpha=0.6)

plt.title("Emotion Confidence Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Confidence")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 3. Pie chart of emotions ---
plt.figure(figsize=(6,6))
plt.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Emotion Distribution")
plt.show()
