# python emonarrify/phase1/inference.py
from model import generate_story
from datasets import load_dataset
from model import generate_story
import os
import json

cnt = 0
total = 0

JSONL_PATH = "data/phase1_vist_train_new.jsonl"
OUTPUT_PATH = "predictions_train.jsonl"


dataset = load_dataset("json", data_files=JSONL_PATH)["train"]
dataset = dataset.filter(lambda x: os.path.exists(os.path.expanduser(x["image_path"])))

with open(OUTPUT_PATH, "w") as f:
    for sample in dataset:
        result = generate_story(sample["image_path"], sample["emotion_label"])
        if result["emotion_label"] == sample["emotion_label"]:
            cnt += 1
        
        total += 1

        f.write(json.dumps(result) + "\n")


print("Accuracy: ")
print(cnt / total)
