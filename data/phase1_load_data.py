import json
import requests
import time
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import pipeline


INPUT_JSON = "sis/test.story-in-sequence.json"
OUTPUT_JSONL = "phase1_vist.jsonl"
IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)

NUM_SAMPLES = 10 


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

def download_image(url, save_path):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(r.content)

        return True
    except Exception as e:
        print(f"Download failed: {url} -> {e}")
        return False

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

    print(f"Total flat annotations: {len(flat_annotations)}")
    print(f"Total grouped stories: {len(stories)}")
    print(f"Total indexed images: {len(image_map)}")

    return stories, image_map

def build_full_story(story_items):
    texts = []
    for x in story_items:
        text = x.get("text", "").strip()
        if text:
            texts.append(text)
    return " ".join(texts).strip()

def get_mid_image_info(story_items, image_map):
    if not story_items:
        return None, None, None

    mid = story_items[len(story_items) // 2]
    photo_id = str(mid.get("photo_flickr_id"))

    if not photo_id or photo_id == "None":
        return None, None, None

    img_info = image_map.get(photo_id)
    if not img_info:
        return photo_id, None, None

    url = img_info.get("url_o") or img_info.get("url_m") or img_info.get("url")
    return photo_id, url, img_info

def select_best_story_per_image(stories, image_map):
    best_by_photo = {}

    for story in stories:
        full_story = build_full_story(story)
        if not full_story:
            continue

        photo_id, url, _ = get_mid_image_info(story, image_map)
        if not photo_id or not url:
            continue

        story_len = len(full_story.split())

        # keep longest story for same image
        if photo_id not in best_by_photo or story_len > best_by_photo[photo_id]["story_len"]:
            best_by_photo[photo_id] = {
                "photo_id": photo_id,
                "url": url,
                "story_items": story,
                "full_story": full_story,
                "story_len": story_len,
            }

    selected = list(best_by_photo.values())
    selected = sorted(selected, key=lambda x: x["story_len"], reverse=True)

    print(f"Unique downloadable candidates: {len(selected)}")
    return selected

def build_dataset():
    stories, image_map = load_vist_sis()

    selected_samples = select_best_story_per_image(stories, image_map)

    clf = load_emotion_clf()
    results = []

    print("Processing selected stories...")

    for i, sample in enumerate(tqdm(selected_samples)):
        if len(results) >= NUM_SAMPLES:
            break

        try:
            photo_id = sample["photo_id"]
            url = sample["url"]
            full_story = sample["full_story"]

            if not full_story:
                continue

            save_path = IMAGE_DIR / f"{photo_id}.jpg"

            if not save_path.exists():
                success = download_image(url, save_path)
                if not success:
                    continue
                time.sleep(0.1)

            outputs = clf(full_story)[0]
            emotion = map_emotion(outputs)

            record = {
                "image_path": str(save_path),
                "narrative_text": full_story,
                "emotion_label": emotion
            }
            results.append(record)

        except Exception as e:
            print(f"Error at sample {i}: {e}")
            continue

    save_jsonl(results)

    if len(results) < NUM_SAMPLES:
        print(
            f"Warning: requested {NUM_SAMPLES} samples, "
            f"but only got {len(results)} successful samples."
        )

def save_jsonl(data):
    print("Saving JSONL...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done! Saved {len(data)} samples to {OUTPUT_JSONL}")

if __name__ == "__main__":
    build_dataset()