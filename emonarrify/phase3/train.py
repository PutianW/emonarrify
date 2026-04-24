"""
Phase 3 training (stub stage).

Current goal:
  - Validate Phase 3 training manifest
  - Initialize learnable emotion lookup table
  - Export lookup table JSON for Phase 2 alignment/fallback

This file is intentionally lightweight for MVP integration before full TTS training.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from ..config import EMOTION_LABELS, LOOKUP_TABLE_PATH
from .model import EmotionLookupTable


REQUIRED_MANIFEST_KEYS = {
    "utterance_id",
    "speaker_id",
    "audio_path",
    "narrative_text",
    "emotion_label",
    "split",
}


@dataclass
class Phase3StubTrainResult:
    manifest_path: str
    num_records: int
    split_counts: Dict[str, int]
    emotion_counts: Dict[str, int]
    lookup_table_path: str


def _read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_idx} in {path}: {exc}") from exc
            records.append(item)
    return records


def _validate_manifest(records: List[dict], manifest_path: Path) -> None:
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    for i, rec in enumerate(records, start=1):
        missing = REQUIRED_MANIFEST_KEYS - set(rec.keys())
        if missing:
            raise ValueError(f"Record #{i} missing keys {sorted(missing)} in {manifest_path}")

        label = str(rec["emotion_label"]).strip().lower()
        if label not in EMOTION_LABELS:
            raise ValueError(
                f"Record #{i} has invalid emotion_label={rec['emotion_label']!r}. "
                f"Expected one of {EMOTION_LABELS}."
            )

        split = str(rec["split"]).strip().lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Record #{i} has invalid split={rec['split']!r}. Expected train/val/test."
            )

        wav_path = Path(str(rec["audio_path"])).expanduser()
        if not wav_path.exists():
            raise FileNotFoundError(f"Record #{i} audio file not found: {wav_path}")


def train_phase3_stub(
    manifest_path: str,
    lookup_out_path: str = LOOKUP_TABLE_PATH,
    seed: int = 42,
) -> Phase3StubTrainResult:
    """Run MVP stub training for Phase 3 and export lookup table JSON."""
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    records = _read_jsonl(manifest)
    _validate_manifest(records, manifest)

    split_counter = Counter(str(r["split"]).strip().lower() for r in records)
    emotion_counter = Counter(str(r["emotion_label"]).strip().lower() for r in records)

    # Stub initialization: deterministic lookup table init.
    torch.manual_seed(seed)
    lookup = EmotionLookupTable()
    torch.nn.init.normal_(lookup.embedding.weight, mean=0.0, std=0.02)

    out_path = Path(lookup_out_path).expanduser().resolve()
    lookup.export_json(str(out_path))

    result = Phase3StubTrainResult(
        manifest_path=str(manifest),
        num_records=len(records),
        split_counts={"train": split_counter["train"], "val": split_counter["val"], "test": split_counter["test"]},
        emotion_counts={label: emotion_counter[label] for label in EMOTION_LABELS},
        lookup_table_path=str(out_path),
    )

    print("[Phase3][TrainStub] Manifest:", result.manifest_path)
    print("[Phase3][TrainStub] Records:", result.num_records)
    print("[Phase3][TrainStub] Split counts:", result.split_counts)
    print("[Phase3][TrainStub] Emotion counts:", result.emotion_counts)
    print("[Phase3][TrainStub] Exported lookup table:", result.lookup_table_path)

    return result