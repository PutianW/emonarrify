import json
import requests
import time
from collections import defaultdict
from tqdm import tqdm
from transformers import pipeline

import boto3
from io import BytesIO


INPUT_JSON = "sis/val.story-in-sequence.json"
OUTPUT_JSONL = "phase1_vist_val.jsonl"

MAX_PER_EMOTION = 4000
MAX_TOTAL = 20000

S3_BUCKET = "emonarrify-cassie"
s3 = boto3.client("s3")

def load_emotion_clf():
    print("Loading emotion classifier...")
    clf = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
        device=-1
    )
    return clf


def map_emotion(labels):
    label_scores = {l["label"]: l["score"] for l in labels}
    top_label = max(label_scores, key=label_scores.get)

    if top_label in ["joy", "amusement", "love", "excitement", "approval", "optimism", "admiration"]:
        return "happy"
    elif top_label in ["anger", "annoyance", "disapproval"]:
        return "angry"
    elif top_label in ["sadness", "grief", "disappointment", "fear", "remorse"]:
        return "sad"
    elif top_label in ["surprise", "realization"]:
        return "surprise"
    else:
        return "neutral"

def download_and_upload_image(url, photo_id):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()

        img_bytes = BytesIO(r.content)
        s3_key = f"images/val/{photo_id}.jpg"

        s3.upload_fileobj(img_bytes, S3_BUCKET, s3_key)

        return f"s3://{S3_BUCKET}/{s3_key}"

    except Exception as e:
        print(f"Download/upload failed: {url} -> {e}")
        return None

def load_vist_sis():
    print("Loading VIST SIS JSON...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_annotations = data.get("annotations", [])
    raw_images = data.get("images", [])

    image_map = {}

    for item in raw_images:
        if isinstance(item, dict) and "images" in item:
            for img in item["images"]:
                img_id = str(img.get("id"))
                if img_id and img_id != "None":
                    image_map[img_id] = img
        elif isinstance(item, dict):
            img_id = str(item.get("id"))
            if img_id and img_id != "None":
                image_map[img_id] = item

    flat_annotations = []
    for item in raw_annotations:
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict):
                    flat_annotations.append(sub)
        elif isinstance(item, dict):
            flat_annotations.append(item)

    grouped = defaultdict(list)
    for ann in flat_annotations:
        story_id = ann.get("story_id")
        if story_id is None:
            continue
        grouped[str(story_id)].append(ann)

    stories = []
    for story_id, items in grouped.items():
        items = sorted(items, key=lambda x: x.get("worker_arranged_photo_order", 0))
        stories.append(items)

    print(f"Total stories: {len(stories)}")
    print(f"Total images: {len(image_map)}")

    return stories, image_map

def build_image_text_best(stories, image_map):
    best_by_photo = {}

    for story in stories:
        for ann in story:
            text = ann.get("text", "").strip()
            photo_id = str(ann.get("photo_flickr_id"))

            if not text or not photo_id or photo_id == "None":
                continue

            # if the sentence is too short
            if len(text.split()) < 5:
                continue

            img_info = image_map.get(photo_id)
            if not img_info:
                continue

            url = img_info.get("url_o") or img_info.get("url_m") or img_info.get("url")
            if not url:
                continue

            text_len = len(text.split())

            if (photo_id not in best_by_photo or 
                text_len > best_by_photo[photo_id]["text_len"]):

                best_by_photo[photo_id] = {
                    "photo_id": photo_id,
                    "url": url,
                    "text": text,
                    "text_len": text_len
                }

    samples = list(best_by_photo.values())
    print(f"Unique images (best text): {len(samples)}")
    return samples

def save_jsonl(data):
    print("Saving JSONL...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} samples")


def upload_jsonl_to_s3():
    print("Uploading JSONL to S3...")
    s3.upload_file(OUTPUT_JSONL, S3_BUCKET, OUTPUT_JSONL)
    print("Upload done!")

def build_dataset():
    stories, image_map = load_vist_sis()
    selected_samples = build_image_text_best(stories, image_map)

    clf = load_emotion_clf()
    results = []
    emotion_counts = defaultdict(int)

    print("Processing samples...")

    for i, sample in enumerate(tqdm(selected_samples)):
        if len(results) >= MAX_TOTAL:
            break

        try:
            photo_id = sample["photo_id"]
            url = sample["url"]
            text = sample["text"]

            image_s3_path = download_and_upload_image(url, photo_id)
            if not image_s3_path:
                continue

            outputs = clf(text)[0]
            emotion = map_emotion(outputs)

            if emotion_counts[emotion] >= MAX_PER_EMOTION:
                continue

            record = {
                "image_path": image_s3_path,
                "narrative_text": text,
                "emotion_label": emotion
            }

            results.append(record)
            emotion_counts[emotion] += 1

            time.sleep(0.1)

        except Exception as e:
            print(f"Error at {i}: {e}")
            continue

    save_jsonl(results)
    upload_jsonl_to_s3()

    print(f"Done! Total samples: {len(results)}")
    print("Emotion distribution:", dict(emotion_counts))


if __name__ == "__main__":
    build_dataset()