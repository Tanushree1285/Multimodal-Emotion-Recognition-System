import os
import csv

# === PATHS (according to your structure) ===
DATA_ROOT = "../data/audio"
OUTPUT_FILE = "../manifests/audio_manifest.csv"

# === Emotion Mappings ===
RAVDESS_EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}

CREMAD_EMOTIONS = {
    "neu": "neutral",
    "cal": "calm",
    "hap": "happy",
    "sad": "sad",
    "ang": "angry",
    "fea": "fearful",
    "dis": "disgust",
    "sur": "surprised"
}


def parse_ravdess(filename):
    """Extract emotion label from RAVDESS file name."""
    try:
        parts = filename.split("-")
        emotion_id = int(parts[2])
        return RAVDESS_EMOTIONS.get(emotion_id)
    except Exception:
        return None


def parse_cremad(filename):
    """Extract emotion label from CREMA-D file name."""
    try:
        parts = filename.split("_")
        emotion_code = parts[2].lower()
        return CREMAD_EMOTIONS.get(emotion_code)
    except Exception:
        return None


def main():
    rows = [("path", "label")]

    for root, _, files in os.walk(DATA_ROOT):
        for f in files:
            if not f.endswith(".wav"):
                continue

            full_path = os.path.join(root, f).replace("\\", "/")

            if "ravdess" in full_path.lower():
                label = parse_ravdess(f)
            elif "creamad" in full_path.lower():
                label = parse_cremad(f)
            else:
                label = None

            if label:
                rows.append((full_path, label))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

    print(f"✅ Manifest created at: {OUTPUT_FILE}")
    print(f"✅ Total valid samples: {len(rows)-1}")


if __name__ == "__main__":
    main()
