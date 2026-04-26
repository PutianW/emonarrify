"""End-to-end emotion-expression evaluation via SER classifier.

Pipeline: image -> EmoNarrifyPipeline.run() -> wav -> SER -> predicted emotion.
Compare predicted emotion vs ground-truth emotion label from phase1_holdout
to compute 5x5 confusion matrix + per-class accuracy.

Two-pass: emona env has VITS+clip (Pipeline) but no transformers (SER);
emona_ser env has transformers but no VITS+clip. Run twice:

    /home/ubuntu/miniconda3/envs/emona/bin/python      scripts/eval_e2e_ser.py --mode synthesize
    /home/ubuntu/miniconda3/envs/emona_ser/bin/python  scripts/eval_e2e_ser.py --mode classify

Pass 1 (synthesize): writes 56 wavs to outputs/ser_e2e_wavs/<idx>_<gt>.wav
                     and outputs/ser_e2e_pipeline.jsonl (per-image metadata).
Pass 2 (classify):   reads the wavs + pipeline.jsonl, runs SER on each, writes
                     outputs/ser_e2e_eval.json with confusion matrix + metrics.

SER caveat: superb/wav2vec2-base-superb-er emits 4-class IEMOCAP labels
(neu / hap / ang / sad). Our 5-class space includes 'surprise', which has no
SER counterpart. Surprise utterances will route to the nearest 4-class label
(usually hap or ang). Surfaced in the eval JSON so the limitation is visible
to anyone reading the result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_JSONL = ROOT / "data/splits/phase1_holdout.jsonl"

EMOTION_LABELS = ["neutral", "happy", "angry", "sad", "surprise"]
SER_TO_OUR = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}


def _paths(fallback: bool):
    """Return (wavs_dir, pipeline_jsonl, eval_json) tuned for the run mode."""
    suffix = "_fallback" if fallback else ""
    return (
        ROOT / f"outputs/ser_e2e_wavs{suffix}",
        ROOT / f"outputs/ser_e2e_pipeline{suffix}.jsonl",
        ROOT / f"outputs/ser_e2e_eval{suffix}.json",
    )


def _resolve_image_path(rec: dict) -> str | None:
    """Prefer resolved_image_path; fall back to image_path; verify exists."""
    for key in ("resolved_image_path", "image_path"):
        p = rec.get(key)
        if p and os.path.exists(p):
            return p
    return None


def _iter_holdout():
    with open(HOLDOUT_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def cmd_synthesize(fallback: bool):
    sys.path.insert(0, str(ROOT))
    from emonarrify import EmoNarrifyPipeline
    import numpy as np
    from PIL import Image
    import soundfile as sf

    wavs_dir, pipeline_jsonl, _ = _paths(fallback)
    print(f"[synthesize] loading EmoNarrifyPipeline (fallback={fallback}) ...")
    pipe = EmoNarrifyPipeline(
        phase1_adapter=None,
        phase2_mlp=str(ROOT / "weights/phase2_mlp.pt"),
        phase3_tts=str(ROOT / "weights/phase3_new_v1/G_18000.pth"),
        use_fallback=fallback,
    )

    wavs_dir.mkdir(parents=True, exist_ok=True)
    pipeline_records = []
    n_total = n_skipped = 0
    gt_dist = Counter()

    for i, rec in enumerate(_iter_holdout()):
        n_total += 1
        gt_label = rec.get("emotion_label")
        gt_dist[gt_label] += 1
        img_path = _resolve_image_path(rec)
        if img_path is None:
            print(f"[synthesize] [WARN] image not found, skip: {rec.get('image_path')}")
            n_skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            result = pipe.run(img)
        except Exception as e:
            print(f"[synthesize] [WARN] pipeline.run failed on idx={i}: {e}")
            n_skipped += 1
            continue

        audio = result["audio"].astype(np.float32)
        sr = int(result["sample_rate"])
        wav_path = wavs_dir / f"{i:03d}_{gt_label}.wav"
        sf.write(str(wav_path), audio, sr)

        pipeline_records.append({
            "idx": i,
            "image_path": img_path,
            "gt_label": gt_label,
            "wav_path": str(wav_path.relative_to(ROOT)),
            "sample_rate": sr,
            "audio_duration_s": float(len(audio) / sr),
            "audio_peak": float(np.abs(audio).max()),
            "audio_rms": float((audio ** 2).mean() ** 0.5),
            "embedding_source": result.get("embedding_source"),
            "pipeline_emotion": result.get("emotion_label"),
            "pipeline_narrative": result.get("narrative_text"),
            "parse_confidence": result.get("parse_confidence"),
        })
        if (i + 1) % 10 == 0:
            print(f"[synthesize] {i + 1} done")

    with open(pipeline_jsonl, "w") as f:
        for r in pipeline_records:
            f.write(json.dumps(r) + "\n")

    print(f"[synthesize] total={n_total} synthesized={len(pipeline_records)} "
          f"skipped={n_skipped}")
    print(f"[synthesize] gt_distribution={dict(gt_dist)}")
    print(f"[synthesize] wavs_dir={wavs_dir}")
    print(f"[synthesize] pipeline_jsonl={pipeline_jsonl}")


def cmd_classify(fallback: bool):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import pipeline as hf_pipeline

    _, pipeline_jsonl, eval_json = _paths(fallback)
    if not pipeline_jsonl.exists():
        sys.exit(f"[classify] missing {pipeline_jsonl}; run --mode synthesize first")

    device = 0 if torch.cuda.is_available() else -1
    print(f"[classify] loading SER (superb/wav2vec2-base-superb-er) device={device}")
    ser = hf_pipeline(
        "audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=device,
    )

    samples = []
    confusion: dict[str, Counter] = {gt: Counter() for gt in EMOTION_LABELS}

    with open(pipeline_jsonl) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[classify] {len(records)} wavs to classify")
    for i, rec in enumerate(records):
        wav_path = ROOT / rec["wav_path"]
        audio, sr = sf.read(str(wav_path), dtype="float32")
        ser_input = {"array": audio, "sampling_rate": sr}
        ser_output = ser(ser_input, top_k=4)

        top_raw = ser_output[0]["label"].lower()
        top_score = float(ser_output[0]["score"])
        mapped = SER_TO_OUR.get(top_raw, "unknown")
        gt = rec["gt_label"]
        confusion[gt][mapped] += 1

        samples.append({
            **rec,
            "ser_top1_raw": top_raw,
            "ser_top1_mapped": mapped,
            "ser_top1_score": top_score,
            "ser_full_output": [
                {"label": o["label"], "score": float(o["score"])} for o in ser_output
            ],
        })
        if (i + 1) % 10 == 0:
            print(f"[classify] {i + 1}/{len(records)} done")

    n_correct = sum(1 for s in samples if s["ser_top1_mapped"] == s["gt_label"])
    n_total = len(samples)
    overall_acc = n_correct / n_total if n_total else 0.0

    per_class_acc: dict[str, dict] = {}
    for gt in EMOTION_LABELS:
        in_class = [s for s in samples if s["gt_label"] == gt]
        n = len(in_class)
        c = sum(1 for s in in_class if s["ser_top1_mapped"] == gt)
        per_class_acc[gt] = {"n": n, "correct": c, "acc": (c / n) if n else None}
    valid_accs = [v["acc"] for v in per_class_acc.values() if v["acc"] is not None]
    macro_acc = (sum(valid_accs) / len(valid_accs)) if valid_accs else 0.0

    pipeline_desc = (
        "EmoNarrifyPipeline use_fallback=True "
        "(Phase 2 bypassed, Phase 3 driven by lookup_table[Phase 1 emotion_label])"
        if fallback
        else "EmoNarrifyPipeline (stub Phase 1 + Phase 2 v3 + Run 2 G_18000)"
    )
    output = {
        "config": {
            "pipeline": pipeline_desc,
            "fallback_mode": fallback,
            "ser_model": "superb/wav2vec2-base-superb-er",
            "ser_label_map": SER_TO_OUR,
            "holdout_jsonl": str(HOLDOUT_JSONL.relative_to(ROOT)),
            "ser_caveat": "4-class IEMOCAP (neu/hap/ang/sad); 'surprise' GT routes to "
                          "nearest 4-class label since SER has no surprise output node.",
        },
        "n_total": n_total,
        "overall_accuracy": overall_acc,
        "macro_accuracy": macro_acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": {gt: dict(counter) for gt, counter in confusion.items()},
        "samples": samples,
    }

    with open(eval_json, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"=== SER e2e eval ===")
    print(f"overall_acc: {overall_acc:.4f} ({n_correct}/{n_total})")
    print(f"macro_acc:   {macro_acc:.4f}")
    print(f"per-class:")
    for gt, v in per_class_acc.items():
        if v["acc"] is None:
            print(f"  {gt:>9s}: N=0")
        else:
            print(f"  {gt:>9s}: {v['correct']}/{v['n']} = {v['acc']:.3f}")
    print()
    cols = EMOTION_LABELS + ["unknown"]
    print(f"confusion (true rows x pred cols):")
    print("        " + " ".join(f"{c:>9s}" for c in cols))
    for gt in EMOTION_LABELS:
        row = [confusion[gt].get(c, 0) for c in cols]
        print(f"{gt:>8s}" + " ".join(f"{n:>9d}" for n in row))
    print()
    print(f"output: {eval_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthesize", "classify"], required=True)
    ap.add_argument("--fallback", action="store_true",
                    help="Run with use_fallback=True (Phase 2 bypassed, lookup-only).")
    args = ap.parse_args()
    if args.mode == "synthesize":
        cmd_synthesize(args.fallback)
    else:
        cmd_classify(args.fallback)


if __name__ == "__main__":
    main()
