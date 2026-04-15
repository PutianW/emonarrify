"""
Phase 3 data utilities.

This module provides the first implementation stage for Phase 3:
1) Read ESD directory structure
2) Parse transcript files
3) Normalize emotion labels to project-standard labels
4) Build train/val/test JSONL manifest for TTS training
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..config import DATA_DIR, EMOTION_LABELS


EMOTION_FOLDER_BY_LABEL = {
    "neutral": "Neutral",
    "happy": "Happy",
    "angry": "Angry",
    "sad": "Sad",
    "surprise": "Surprise",
}

_EMOTION_NORMALIZE = {
    # canonical
    "neutral": "neutral",
    "happy": "happy",
    "angry": "angry",
    "sad": "sad",
    "surprise": "surprise",
    # ESD folder/title case
    "Neutral": "neutral",
    "Happy": "happy",
    "Angry": "angry",
    "Sad": "sad",
    "Surprise": "surprise",
    # Chinese labels observed in ESD transcripts
    "中立": "neutral",
    "快乐": "happy",
    "生气": "angry",
    "伤心": "sad",
    "惊喜": "surprise",
}


@dataclass(frozen=True)
class ESDRecord:
    """One normalized Phase 3 training item."""

    utterance_id: str
    speaker_id: str
    audio_path: str
    narrative_text: str
    emotion_label: str
    split: str


def normalize_emotion_label(raw_label: str) -> str:
    """Map raw ESD label (Chinese/English) to project-standard emotion label."""
    if raw_label in _EMOTION_NORMALIZE:
        return _EMOTION_NORMALIZE[raw_label]

    lowered = raw_label.strip().lower()
    if lowered in _EMOTION_NORMALIZE:
        return _EMOTION_NORMALIZE[lowered]

    raise ValueError(
        f"Unknown emotion label: {raw_label!r}. "
        f"Expected one of {sorted(set(_EMOTION_NORMALIZE.keys()))}."
    )


def _iter_speaker_dirs(esd_root: Path) -> Iterable[Path]:
    for child in sorted(esd_root.iterdir()):
        if child.is_dir() and child.name.isdigit():
            yield child


def _parse_transcript_line(line: str) -> Tuple[str, str, str]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        raise ValueError(f"Malformed transcript line: {line!r}")
    utterance_id = parts[0].strip()
    narrative_text = parts[1].strip()
    raw_emotion = parts[2].strip()
    return utterance_id, narrative_text, raw_emotion


def _resolve_audio_path(speaker_dir: Path, utterance_id: str, emotion_label: str) -> Optional[Path]:
    """Resolve utterance wav path; return None if not found."""
    canonical_folder = EMOTION_FOLDER_BY_LABEL[emotion_label]
    expected = speaker_dir / canonical_folder / f"{utterance_id}.wav"
    if expected.exists():
        return expected

    # Fallback scan in case data folder names differ.
    for folder in EMOTION_FOLDER_BY_LABEL.values():
        candidate = speaker_dir / folder / f"{utterance_id}.wav"
        if candidate.exists():
            return candidate
    return None


def _speaker_split_map(
    speaker_ids: List[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, str]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in [0,1), got {test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    speakers = list(speaker_ids)
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    # Keep at least one train speaker when possible.
    while n_test + n_val >= n and n > 1:
        if n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    test_speakers = set(speakers[:n_test])
    val_speakers = set(speakers[n_test : n_test + n_val])

    split_map: Dict[str, str] = {}
    for sid in speakers:
        if sid in test_speakers:
            split_map[sid] = "test"
        elif sid in val_speakers:
            split_map[sid] = "val"
        else:
            split_map[sid] = "train"
    return split_map


def build_phase3_esd_manifest(
    esd_root: str,
    output_jsonl: Optional[str] = None,
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> List[ESDRecord]:
    """
    Build a normalized Phase 3 manifest from ESD.

    Output schema (one JSON per line):
      {
        "utterance_id": str,
        "speaker_id": str,
        "audio_path": str,
        "narrative_text": str,
        "emotion_label": str,   # one of project EMOTION_LABELS
        "split": "train" | "val" | "test"
      }
    """
    root = Path(esd_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Invalid ESD root directory: {root}")

    speaker_dirs = list(_iter_speaker_dirs(root))
    if not speaker_dirs:
        raise RuntimeError(f"No speaker directories found under: {root}")

    split_by_speaker = _speaker_split_map(
        speaker_ids=[p.name for p in speaker_dirs],
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    all_records: List[ESDRecord] = []
    skipped_missing_wav = 0

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        transcript_path = speaker_dir / f"{speaker_id}.txt"
        if not transcript_path.exists():
            raise FileNotFoundError(f"Missing transcript file: {transcript_path}")

        split = split_by_speaker[speaker_id]

        with transcript_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                if not raw_line.strip():
                    continue
                utterance_id, narrative_text, raw_emotion = _parse_transcript_line(raw_line)
                emotion_label = normalize_emotion_label(raw_emotion)
                if emotion_label not in EMOTION_LABELS:
                    raise ValueError(
                        f"Emotion {emotion_label!r} not in project labels {EMOTION_LABELS}."
                    )

                wav_path = _resolve_audio_path(speaker_dir, utterance_id, emotion_label)
                if wav_path is None:
                    skipped_missing_wav += 1
                    continue

                all_records.append(
                    ESDRecord(
                        utterance_id=utterance_id,
                        speaker_id=speaker_id,
                        audio_path=str(wav_path),
                        narrative_text=narrative_text,
                        emotion_label=emotion_label,
                        split=split,
                    )
                )

    if output_jsonl is None:
        output_path = Path(DATA_DIR) / "phase3_esd_manifest.jsonl"
    else:
        output_path = Path(output_jsonl).expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")

    split_counts = {"train": 0, "val": 0, "test": 0}
    emotion_counts = {label: 0 for label in EMOTION_LABELS}
    for record in all_records:
        split_counts[record.split] += 1
        emotion_counts[record.emotion_label] += 1

    print(f"[Phase3][Data] ESD root: {root}")
    print(f"[Phase3][Data] Speakers: {len(speaker_dirs)}")
    print(f"[Phase3][Data] Records: {len(all_records)}")
    print(f"[Phase3][Data] Split counts: {split_counts}")
    print(f"[Phase3][Data] Emotion counts: {emotion_counts}")
    if skipped_missing_wav:
        print(f"[Phase3][Data] Skipped missing wav: {skipped_missing_wav}")
    print(f"[Phase3][Data] Manifest saved: {output_path}")

    return all_records