"""
Phase 1: Vision-to-Text with Emotion (VLM Instruction Tuning)
Owner: Cassie Chang, cassiech
"""

import os
import json
import requests
import time
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from transformers import pipeline

# =========================
# CONFIG
# =========================
NUM_SAMPLES = 10
SAVE_DIR = "images"
OUTPUT_JSONL = "phase1_data.jsonl"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Load VIST via HF API (no dataset script)
# =========================
def load_VIST(num_samples):
    print("Loading dataset from HF API...")

    url = "https://datasets-server.huggingface.co/rows"

    params = {
        "dataset": "sil-ai/bloom-vist",
        "config": "eng",
        "split": "train",
        "offset": 0,
        "length": num_samples
    }

    r = requests.get(url, params=params)
    data = r.json()

    samples = []
    for row in data["rows"]:
        samples.append(row["row"])

    return samples

# =========================
# Emotion classifier
# =========================
def load_emotion_clf(model_name="SamLowe/roberta-base-go_emotions", top_k=None):
    print("Loading emotion classifier...")
    clf = pipeline(
        "text-classification",
        model=model_name,
        top_k=top_k
    )
    return clf

# =========================
# Emotion mapping
# =========================
def map_emotion(labels):
    label_scores = {l["label"]: l["score"] for l in labels}
    top_label = max(label_scores, key=label_scores.get)

    if top_label in ["joy", "amusement", "love"]:
        return "happy"
    elif top_label in ["anger", "annoyance"]:
        return "angry"
    elif top_label in ["sadness", "grief", "disappointment", "fear"]:
        return "sad"
    elif top_label in ["surprise"]:
        return "surprise"
    else:
        return "neutral"

# =========================
# Download image
# =========================
def download_image(url, path):
    try:
        r = requests.get(url, timeout=10)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(path)
        return True
    except Exception as e:
        print(f"Image download failed: {e}")
        return False

# =========================
# Main pipeline
# =========================
def phase1_load_dataset():
    dataset = load_VIST(NUM_SAMPLES)
    clf = load_emotion_clf()

    data = []

    print("Processing dataset...")

    for i, sample in enumerate(tqdm(dataset)):
        try:
            story_list = sample["story"]

            texts = [s["text"] for s in story_list if s["text"].strip() != ""]
            if len(texts) == 0:
                continue

            story = " ".join(texts)

            img_url = story_list[0]["image_url"]
            img_path = os.path.join(SAVE_DIR, f"{i}.jpg")

            success = download_image(img_url, img_path)
            if not success:
                continue

            # emotion prediction
            outputs = clf(story)[0]
            emotion = map_emotion(outputs)

            record = {
                "image_path": img_path,
                "narrative_text": story,
                "emotion_label": emotion
            }

            data.append(record)

            # 避免 API 太快被限制
            time.sleep(0.1)

        except Exception as e:
            print(f"Error at {i}: {e}")
            continue

    save(data)

# =========================
# Save jsonl
# =========================
def save(data):
    print("Saving jsonl...")

    with open(OUTPUT_JSONL, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Done! Saved {len(data)} samples.")

# =========================
# Run
# =========================
if __name__ == "__main__":
    phase1_load_dataset()