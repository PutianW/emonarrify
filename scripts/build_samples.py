"""Build 5-sample exhibit set covering all five emotion classes.

Pipeline: stub Phase 1 + Phase 2 v3 + Phase 3 Run 1 (deployed, Path A).
Image selection: deterministic first-match per emotion in data/splits/test.jsonl,
deduped against the commit 20 demo set (outputs/demo/manifest.json), with
data/splits/val.jsonl fallback if the test split has no eligible image.
Manifest annotates GoEmotions GT label, Phase 2 NN-argmax label, and Phase 2
embedding cos sim per lookup vector for paper-figure use.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_JSONL = ROOT / "data/splits/test.jsonl"
VAL_JSONL = ROOT / "data/splits/val.jsonl"
DEMO_MANIFEST = ROOT / "outputs/demo/manifest.json"
OUT_DIR = ROOT / "outputs/samples"
LOOKUP_JSON = ROOT / "weights/emotion_lookup_table.json"

EMOTION_LABELS = ["neutral", "happy", "angry", "sad", "surprise"]


def _resolve(rec):
    """Return (image_id, abs_path_or_None)."""
    image_id = os.path.splitext(os.path.basename(rec["image_path"]))[0]
    p = rec.get("resolved_image_path") or rec["image_path"]
    return image_id, (p if os.path.exists(p) else None)


def _select_one(emotion, records, used_ids):
    for r in records:
        if r["emotion_label"] != emotion:
            continue
        image_id, p = _resolve(r)
        if image_id in used_ids or p is None:
            continue
        return {**r, "image_id": image_id, "image_path_resolved": p}
    return None


def main():
    sys.path.insert(0, str(ROOT))
    import torch
    import torch.nn.functional as F
    import soundfile as sf
    from PIL import Image
    from emonarrify import EmoNarrifyPipeline

    used_ids = set()
    with open(DEMO_MANIFEST) as f:
        used_ids = {s["image_id"] for s in json.load(f)["samples"]}
    print(f"[select] demo dedup set: {len(used_ids)} ids")

    test_records = [json.loads(l) for l in open(TEST_JSONL)]
    val_records = [json.loads(l) for l in open(VAL_JSONL)]

    selected = []
    fallback_emotions = []
    for emotion in EMOTION_LABELS:
        rec = _select_one(emotion, test_records, used_ids)
        split_used = "test"
        if rec is None:
            rec = _select_one(emotion, val_records, used_ids)
            split_used = "val"
            if rec is not None:
                fallback_emotions.append(emotion)
        if rec is None:
            print(f"[select] ERROR: no image for {emotion} in test or val")
            continue
        rec["target_emotion"] = emotion
        rec["split_used"] = split_used
        selected.append(rec)
        print(f"[select] {emotion:>9s}: {rec['image_id']} (split={split_used})")
    if fallback_emotions:
        print(f"[select] val fallback: {fallback_emotions}")

    pipe = EmoNarrifyPipeline(
        phase1_adapter=None,
        phase2_mlp=str(ROOT / "weights/phase2_mlp.pt"),
        phase3_tts=str(ROOT / "weights/phase3_new_v1/G_18000.pth"),
    )

    with open(LOOKUP_JSON) as f:
        raw = json.load(f)
    lookup_matrix = torch.stack(
        [torch.tensor(raw["embeddings"][l], dtype=torch.float32) for l in EMOTION_LABELS],
        dim=0,
    )
    lookup_n = F.normalize(lookup_matrix, dim=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, rec in enumerate(selected):
        img = Image.open(rec["image_path_resolved"]).convert("RGB")

        emb = pipe.phase2.encode_image_emotion(img).float()
        emb_n = F.normalize(emb, dim=0)
        cos = (lookup_n @ emb_n).tolist()
        cos_dict = {EMOTION_LABELS[k]: round(float(v), 4) for k, v in enumerate(cos)}
        nn_label = max(cos_dict, key=cos_dict.get)

        result = pipe.run(img)

        wav_name = f"sample_{i:02d}_{rec['target_emotion']}_{rec['image_id']}.wav"
        wav_path = OUT_DIR / wav_name
        sf.write(str(wav_path), result["audio"], result["sample_rate"])

        sample = {
            "sample_index": i,
            "wav_path": str(wav_path.relative_to(ROOT)),
            "image_path": rec["image_path_resolved"],
            "image_id": rec["image_id"],
            "split_source": rec["split_used"],
            "ground_truth_emotion": rec["target_emotion"],
            "ground_truth_narrative": rec.get("narrative_text"),
            "phase1_emotion_returned": result.get("emotion_label"),
            "phase1_narrative_returned": result.get("narrative_text"),
            "phase2_nn_predicted_label": nn_label,
            "phase2_nn_cos_sims": cos_dict,
            "phase2_embedding_norm": round(float(emb.norm()), 4),
            "embedding_source": result.get("embedding_source"),
            "audio_duration_s": round(len(result["audio"]) / result["sample_rate"], 3),
            "audio_peak": round(float(abs(result["audio"]).max()), 4),
            "audio_rms": round(float((result["audio"] ** 2).mean() ** 0.5), 4),
            "audio_sample_rate": int(result["sample_rate"]),
        }
        samples.append(sample)
        print(f"  [{i:02d}] gt={rec['target_emotion']:>9s} "
              f"phase2_nn={nn_label:>9s} dur={sample['audio_duration_s']:.2f}s "
              f"rms={sample['audio_rms']:.4f} peak={sample['audio_peak']:.3f}")

    manifest = {
        "sample_set_description": "Five-sample exhibit, one image per emotion class, "
                                  "deterministic first-match from VIST test split "
                                  "(deduped against commit 20 demo set, val-split "
                                  "fallback if test exhausted).",
        "pipeline_config": {
            "phase1_mode": "stub (canned narrative + neutral label)",
            "phase2_ckpt": "weights/phase2_mlp.pt",
            "phase3_ckpt": "weights/phase3_new_v1/G_18000.pth",
            "lookup_table": "weights/emotion_lookup_table.json",
            "ground_truth_source": "GoEmotions auto-label on VIST narrative",
            "phase2_nn_method": "argmax cos sim (Phase 2 emb vs 5 lookup vectors)",
        },
        "fallback_emotions": fallback_emotions,
        "demo_dedup_count": len(used_ids),
        "samples": samples,
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[done] {len(samples)} wavs + manifest at {OUT_DIR}")


if __name__ == "__main__":
    main()
