"""Phase 3 (new) data preparation: ESD English subset → manifest + 3 VITS filelists.

Reads ESD speaker dirs (English subset 0011-0020), normalizes emotion labels,
filters out missing/short wav files, and writes:
  - data/phase3_esd_manifest_new.jsonl   (one record per utterance, all splits)
  - data/filelists/phase3_new_{train,val,test}.txt  (audio|eid|text per line)

Speaker-out split is hard-coded (Phase 3 spec Q5 decision D):
  train: 0012, 0013, 0014, 0016, 0017, 0018, 0019, 0020   (5M + 3F)
  val:   0011                                              (M, F0=117.7Hz)
  test:  0015                                              (F, F0=211.0Hz)

Audio paths are absolute. ESD lives at /opt/dlami/nvme/ESD (external mount),
relative paths from repo root are awkward across mounts. Re-run this script
if ESD is moved or repo is relocated.

Text is NOT pre-cleaned — VITS english_cleaners2 handles cleaning at training
time inside text_to_sequence(). Manifest text field stays raw for debug.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

from emonarrify.config import EMOTION_LABELS, EMOTION_TO_IDX
from emonarrify.phase3.data import EMOTION_FOLDER_BY_LABEL, normalize_emotion_label


SPLIT_BY_SPEAKER = {
    "0011": "val",
    "0012": "train",
    "0013": "train",
    "0014": "train",
    "0015": "test",
    "0016": "train",
    "0017": "train",
    "0018": "train",
    "0019": "train",
    "0020": "train",
}
ENGLISH_SPEAKERS = sorted(SPLIT_BY_SPEAKER.keys())


def parse_transcript_line(line: str) -> tuple[str, str, str]:
    parts = line.rstrip().split("\t")
    if len(parts) < 3:
        raise ValueError(f"Malformed transcript line: {line!r}")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / w.getframerate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--esd-root", type=str, default="/opt/dlami/nvme/ESD")
    parser.add_argument("--manifest-out", type=str, default="data/phase3_esd_manifest_new.jsonl")
    parser.add_argument("--filelist-dir", type=str, default="data/filelists")
    parser.add_argument("--min-duration", type=float, default=0.3)
    args = parser.parse_args()

    esd_root = Path(args.esd_root).expanduser().resolve()
    manifest_path = Path(args.manifest_out).expanduser().resolve()
    filelist_dir = Path(args.filelist_dir).expanduser().resolve()

    if not esd_root.is_dir():
        raise FileNotFoundError(f"ESD root not found: {esd_root}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    filelist_dir.mkdir(parents=True, exist_ok=True)

    n_speakers_per_split = {"train": 0, "val": 0, "test": 0}
    for sid, split in SPLIT_BY_SPEAKER.items():
        n_speakers_per_split[split] += 1

    print(f"[prepare] ESD root: {esd_root}")
    print(f"[prepare] Speakers: train={n_speakers_per_split['train']} "
          f"val={n_speakers_per_split['val']} test={n_speakers_per_split['test']}")

    records: list[dict] = []
    skipped_missing = 0
    skipped_short = 0

    for sid in ENGLISH_SPEAKERS:
        speaker_dir = esd_root / sid
        transcript_path = speaker_dir / f"{sid}.txt"
        if not transcript_path.is_file():
            raise FileNotFoundError(f"Missing transcript: {transcript_path}")

        split = SPLIT_BY_SPEAKER[sid]
        with transcript_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                if not raw_line.strip():
                    continue
                utt_id, text, raw_emotion = parse_transcript_line(raw_line)
                emotion_label = normalize_emotion_label(raw_emotion)
                if emotion_label not in EMOTION_LABELS:
                    raise ValueError(f"Unknown emotion {emotion_label!r} after normalization")

                wav_path = speaker_dir / EMOTION_FOLDER_BY_LABEL[emotion_label] / f"{utt_id}.wav"
                if not wav_path.is_file():
                    skipped_missing += 1
                    continue
                if wav_duration_seconds(wav_path) < args.min_duration:
                    skipped_short += 1
                    continue

                records.append({
                    "utterance_id": utt_id,
                    "speaker_id": sid,
                    "audio_path": str(wav_path),
                    "narrative_text": text,
                    "emotion_label": emotion_label,
                    "emotion_id": EMOTION_TO_IDX[emotion_label],
                    "split": split,
                })

    print(f"[prepare] Skipped (missing wav): {skipped_missing}")
    print(f"[prepare] Skipped (duration < {args.min_duration}s): {skipped_short}")

    split_counts = {"train": 0, "val": 0, "test": 0}
    emotion_counts = {label: 0 for label in EMOTION_LABELS}
    for r in records:
        split_counts[r["split"]] += 1
        emotion_counts[r["emotion_label"]] += 1

    print(f"[prepare] split counts: {split_counts}")
    print(f"[prepare] per-emotion (across all splits): {emotion_counts}")

    with manifest_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prepare] Manifest: {manifest_path}")

    filelist_paths: dict[str, Path] = {
        split: filelist_dir / f"phase3_new_{split}.txt" for split in ("train", "val", "test")
    }
    for split, path in filelist_paths.items():
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                if r["split"] != split:
                    continue
                f.write(f"{r['audio_path']}|{r['emotion_id']}|{r['narrative_text']}\n")
        print(f"[prepare] Filelist ({split}): {path}")


if __name__ == "__main__":
    main()
