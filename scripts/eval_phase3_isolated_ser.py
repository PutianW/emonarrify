"""Phase 3 isolated SER eval: directly drive PatchedVITSBackbone with each
emotion's native lookup vector, on a fixed bank of 11 narratives. Test
whether Phase 3 produces SER-recognizable audio when the input embedding is
exactly the model's own native cluster centroid (i.e. no Phase 2 prediction
noise interposed).

Two-pass for env split (same pattern as eval_e2e_ser.py):

  /home/ubuntu/miniconda3/envs/emona/bin/python      scripts/eval_phase3_isolated_ser.py --mode synthesize
  /home/ubuntu/miniconda3/envs/emona_ser/bin/python  scripts/eval_phase3_isolated_ser.py --mode classify

Outputs:
  outputs/ser_phase3_isolated_wavs/{emotion}_{idx:02d}.wav
  outputs/ser_phase3_isolated_pipeline.jsonl
  outputs/ser_phase3_isolated_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WAVS_DIR = ROOT / "outputs/ser_phase3_isolated_wavs"
PIPELINE_JSONL = ROOT / "outputs/ser_phase3_isolated_pipeline.jsonl"
EVAL_JSON = ROOT / "outputs/ser_phase3_isolated_eval.json"

EMOTION_LABELS = ["neutral", "happy", "angry", "sad", "surprise"]
SER_TO_OUR = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}

NARRATIVES = [
    "A quiet scene unfolds, captured in this image.",
    "We are going to have a presentation tomorrow.",
    "The story begins with a single moment of stillness.",
    "Years pass and memories fade like distant echoes.",
    "Today is the day we have been waiting for.",
    "The road ahead is long and full of possibilities.",
    "Children laughed as the sun set over the meadow.",
    "He walked alone through the empty streets at night.",
    "The discovery changed everything they thought they knew.",
    "Music filled the room and carried us away.",
    "And so the chapter ended, with promises kept.",
]


def cmd_synthesize():
    sys.path.insert(0, str(ROOT))
    from emonarrify.phase3.vits_backbone import PatchedVITSBackbone
    import numpy as np
    import soundfile as sf

    print(f"[synthesize] loading PatchedVITSBackbone ...")
    backbone = PatchedVITSBackbone(
        ckpt_path=str(ROOT / "weights/phase3_new_v1/G_18000.pth"),
    )
    sr = backbone.sampling_rate
    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for emotion in EMOTION_LABELS:
        for i, text in enumerate(NARRATIVES):
            audio = backbone.synthesize_with_label(text, emotion).astype(np.float32)
            wav_path = WAVS_DIR / f"{emotion}_{i:02d}.wav"
            sf.write(str(wav_path), audio, sr)
            records.append({
                "emotion_intended": emotion,
                "narrative_idx": i,
                "narrative": text,
                "wav_path": str(wav_path.relative_to(ROOT)),
                "sample_rate": int(sr),
                "audio_duration_s": float(len(audio) / sr),
                "audio_peak": float(np.abs(audio).max()),
                "audio_rms": float((audio ** 2).mean() ** 0.5),
            })
        print(f"[synthesize] {emotion}: {len(NARRATIVES)} wavs done")

    with open(PIPELINE_JSONL, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[synthesize] total={len(records)} wavs to {WAVS_DIR}")
    print(f"[synthesize] pipeline_jsonl={PIPELINE_JSONL}")


def cmd_classify():
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import pipeline as hf_pipeline

    if not PIPELINE_JSONL.exists():
        sys.exit(f"[classify] missing {PIPELINE_JSONL}; run --mode synthesize first")

    device = 0 if torch.cuda.is_available() else -1
    print(f"[classify] loading SER (superb/wav2vec2-base-superb-er) device={device}")
    ser = hf_pipeline(
        "audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=device,
    )

    samples = []
    confusion: dict[str, Counter] = {gt: Counter() for gt in EMOTION_LABELS}
    with open(PIPELINE_JSONL) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[classify] {len(records)} wavs to classify")
    for i, rec in enumerate(records):
        wav_path = ROOT / rec["wav_path"]
        audio, sr = sf.read(str(wav_path), dtype="float32")
        ser_input = {"array": audio, "sampling_rate": sr}
        ser_output = ser(ser_input, top_k=4)
        top_raw = ser_output[0]["label"].lower()
        mapped = SER_TO_OUR.get(top_raw, "unknown")
        intended = rec["emotion_intended"]
        confusion[intended][mapped] += 1
        samples.append({
            **rec,
            "ser_top1_raw": top_raw,
            "ser_top1_mapped": mapped,
            "ser_top1_score": float(ser_output[0]["score"]),
            "ser_full_output": [
                {"label": o["label"], "score": float(o["score"])} for o in ser_output
            ],
        })

    n_correct = sum(1 for s in samples if s["ser_top1_mapped"] == s["emotion_intended"])
    n_total = len(samples)
    overall = n_correct / n_total if n_total else 0.0

    per_class = {}
    for emo in EMOTION_LABELS:
        in_class = [s for s in samples if s["emotion_intended"] == emo]
        n = len(in_class)
        c = sum(1 for s in in_class if s["ser_top1_mapped"] == emo)
        per_class[emo] = {"n": n, "correct": c, "acc": (c / n) if n else None}

    valid = [v["acc"] for v in per_class.values() if v["acc"] is not None]
    macro = (sum(valid) / len(valid)) if valid else 0.0

    output = {
        "config": {
            "experiment": "Phase 3 isolated SER eval (PatchedVITSBackbone.synthesize_with_label)",
            "narratives_per_emotion": len(NARRATIVES),
            "narratives": NARRATIVES,
            "ser_model": "superb/wav2vec2-base-superb-er",
            "ser_label_map": SER_TO_OUR,
            "ser_caveat": "4-class IEMOCAP; intended 'surprise' has no SER class.",
        },
        "n_total": n_total,
        "overall_accuracy": overall,
        "macro_accuracy": macro,
        "per_class_accuracy": per_class,
        "confusion_matrix": {emo: dict(c) for emo, c in confusion.items()},
        "samples": samples,
    }
    with open(EVAL_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"=== Phase 3 isolated SER eval ===")
    print(f"n_total: {n_total}  overall_acc: {overall:.4f}  macro_acc: {macro:.4f}")
    print(f"per-class:")
    for emo, v in per_class.items():
        if v["acc"] is None:
            print(f"  {emo:>9s}: N=0")
        else:
            print(f"  {emo:>9s}: {v['correct']}/{v['n']} = {v['acc']:.3f}")
    cols = EMOTION_LABELS + ["unknown"]
    print(f"\nconfusion (intended rows x SER pred cols):")
    print("        " + " ".join(f"{c:>9s}" for c in cols))
    for emo in EMOTION_LABELS:
        row = [confusion[emo].get(c, 0) for c in cols]
        print(f"{emo:>8s}" + " ".join(f"{n:>9d}" for n in row))
    print()
    print(f"output: {EVAL_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthesize", "classify"], required=True)
    args = ap.parse_args()
    if args.mode == "synthesize":
        cmd_synthesize()
    else:
        cmd_classify()


if __name__ == "__main__":
    main()
