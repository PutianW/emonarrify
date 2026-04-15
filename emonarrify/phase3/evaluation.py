"""Phase 3 evaluation helpers (MVP scaffold).

This module provides practical first-step evaluation utilities:
1) Manifest distribution summary (for data sanity checks)
2) MOS template generation (for subjective blind tests)

Objective metrics such as MCD/CLIP are intentionally left for the
next milestone when model outputs and references are standardized.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from ..config import EMOTION_LABELS


def summarize_phase3_manifest(manifest_path: str) -> Dict[str, object]:
    """Summarize dataset size, split balance, and emotion balance from JSONL manifest."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    split_counter = Counter()
    emotion_counter = Counter()
    speaker_counter = Counter()
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1

            split = str(rec.get("split", "unknown")).lower()
            split_counter[split] += 1

            emo = str(rec.get("emotion_label", "unknown")).lower()
            emotion_counter[emo] += 1

            speaker = str(rec.get("speaker_id", "unknown"))
            speaker_counter[speaker] += 1

    summary = {
        "manifest_path": str(path),
        "total_records": total,
        "num_speakers": len(speaker_counter),
        "split_counts": {
            "train": split_counter["train"],
            "val": split_counter["val"],
            "test": split_counter["test"],
        },
        "emotion_counts": {label: emotion_counter[label] for label in EMOTION_LABELS},
    }
    return summary


def create_mos_template(audio_items: List[dict], output_csv_path: str) -> str:
    """
    Create MOS scoring template CSV for blind listening tests.

    Each item should include:
      {
        "item_id": str,
        "audio_path": str,
        "emotion_label": str,
        "narrative_text": str
      }
    """
    out_path = Path(output_csv_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "item_id",
        "audio_path",
        "emotion_label",
        "narrative_text",
        "mos_naturalness_1to5",
        "mos_emotion_match_1to5",
        "comments",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in audio_items:
            writer.writerow(
                {
                    "item_id": item.get("item_id", ""),
                    "audio_path": item.get("audio_path", ""),
                    "emotion_label": item.get("emotion_label", ""),
                    "narrative_text": item.get("narrative_text", ""),
                    "mos_naturalness_1to5": "",
                    "mos_emotion_match_1to5": "",
                    "comments": "",
                }
            )

    return str(out_path)