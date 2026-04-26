"""Build demo wav set for the final report.

Selects up to 2 images per emotion from data/splits/test.jsonl (5 emotions x
2 = 10 wavs target; test split has >=3 of each emotion so the target is
always met). Runs the deployed pipeline on each, annotating with both the
GoEmotions GT label and the Phase 2 NN-predicted label (cosine-NN against
the emotion lookup table). Both labels are useful: GT is auto-tagged from
the VIST narrative, so it can disagree with the visual content; the Phase 2
NN label reflects what the image actually projects to.

Outputs:
  outputs/demo/{idx:02d}_{gt}_{image_id}.wav
  outputs/demo/manifest.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "outputs/demo"
TEST_JSONL = ROOT / "data/splits/test.jsonl"
LOOKUP_JSON = ROOT / "weights/emotion_lookup_table.json"

EMOTION_LABELS = ["neutral", "happy", "angry", "sad", "surprise"]


def main():
    sys.path.insert(0, str(ROOT))
    import torch
    import torch.nn.functional as F
    import soundfile as sf
    from PIL import Image
    from emonarrify import EmoNarrifyPipeline

    pipe = EmoNarrifyPipeline(
        phase1_adapter=None,
        phase2_mlp=str(ROOT / "weights/phase2_mlp.pt"),
        phase3_tts=str(ROOT / "weights/phase3_new_v1/G_18000.pth"),
    )

    with open(LOOKUP_JSON) as f:
        raw = json.load(f)
    lookup = {lab: torch.tensor(v, dtype=torch.float32) for lab, v in raw["embeddings"].items()}
    lookup_matrix = torch.stack([lookup[l] for l in EMOTION_LABELS], dim=0)
    lookup_n = F.normalize(lookup_matrix, dim=1)

    by_emotion = defaultdict(list)
    with open(TEST_JSONL) as f:
        for line in f:
            r = json.loads(line)
            by_emotion[r["emotion_label"]].append(r)

    selected = []
    for emo in EMOTION_LABELS:
        selected.extend(by_emotion[emo][:2])
    print(f"[demo] selected {len(selected)} images: "
          f"{[r['emotion_label'] for r in selected]}")

    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    samples = []
    skipped = []
    for i, rec in enumerate(selected):
        img_path = rec.get("resolved_image_path") or rec.get("image_path")
        if not img_path or not os.path.exists(img_path):
            skipped.append({"rec": rec, "reason": f"image not found: {img_path}"})
            print(f"  [{i:02d}] [SKIP] image not found: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")

        phase2_emb = pipe.phase2.encode_image_emotion(img).float()
        emb_n = F.normalize(phase2_emb, dim=0)
        cos_sims = (lookup_n @ emb_n).tolist()
        cos_dict = {EMOTION_LABELS[k]: float(v) for k, v in enumerate(cos_sims)}
        nn_label = max(cos_dict, key=cos_dict.get)

        result = pipe.run(img)

        image_id = os.path.splitext(os.path.basename(img_path))[0]
        wav_name = f"{i:02d}_{rec['emotion_label']}_{image_id}.wav"
        wav_path = DEMO_DIR / wav_name
        sf.write(str(wav_path), result["audio"], result["sample_rate"])

        sample = {
            "wav_path": str(wav_path.relative_to(ROOT)),
            "image_path": img_path,
            "image_id": image_id,
            "ground_truth_emotion": rec["emotion_label"],
            "ground_truth_narrative": rec.get("narrative_text"),
            "phase1_emotion": result.get("emotion_label"),
            "phase2_nn_predicted_label": nn_label,
            "phase2_nn_cos_sims": cos_dict,
            "phase2_embedding_norm": float(phase2_emb.norm()),
            "embedding_source": result.get("embedding_source"),
            "audio_duration_s": float(len(result["audio"]) / result["sample_rate"]),
            "audio_peak": float(abs(result["audio"]).max()),
            "audio_rms": float((result["audio"] ** 2).mean() ** 0.5),
            "narrative_text": result.get("narrative_text"),
            "parse_confidence": result.get("parse_confidence"),
        }
        samples.append(sample)
        print(f"  [{i:02d}] gt={rec['emotion_label']:>9s}  "
              f"phase2_nn={nn_label:>9s}  "
              f"dur={sample['audio_duration_s']:.2f}s  "
              f"peak={sample['audio_peak']:.3f}  "
              f"id={image_id}")

    manifest = {
        "demo_set": f"Phase 1 test split, up to 2 per emotion ({len(samples)} wavs total)",
        "config": {
            "phase1_mode": "stub (canned narrative + neutral label)",
            "phase2_ckpt": "weights/phase2_mlp.pt",
            "phase3_ckpt": "weights/phase3_new_v1/G_18000.pth",
            "lookup_table": "weights/emotion_lookup_table.json",
            "ground_truth_source": "GoEmotions auto-label on VIST narrative",
            "phase2_nn_method": "argmax cos sim (Phase 2 emb vs 5 lookup vectors)",
        },
        "samples": samples,
        "skipped": skipped,
    }
    with open(DEMO_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[demo] {len(samples)} wavs + manifest at {DEMO_DIR}")
    if skipped:
        print(f"[demo] {len(skipped)} skipped (image not found)")


if __name__ == "__main__":
    main()
