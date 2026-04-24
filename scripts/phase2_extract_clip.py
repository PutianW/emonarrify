"""
Phase 2 CLIP feature precompute.

Loads frozen CLIP ViT-B/32, iterates the four splits produced by
data/phase2_build_splits.py (train, val, test, phase1_holdout), preprocesses
and encodes each image, and writes one .pt dict per split:

  data/features/clip_vitb32_{split}.pt  ->  {
      "basenames": list[str]   (length N, order matches features/labels rows)
      "features":  FloatTensor (N, 512), fp32, CPU
      "labels":    list[str]   (length N, emotion_label strings)
  }

phase1_holdout is also cached so future end-to-end pipeline evaluation does
not have to re-run CLIP. It is still excluded from Phase 2 train/val/test.

Sanity checks printed per split: shape, dtype, NaN count, L2-norm mean/std.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

import clip

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
FEATURES_DIR = DATA_DIR / "features"

SPLITS = ["train", "val", "test", "phase1_holdout"]
CLIP_MODEL = "ViT-B/32"


def load_split(split):
    path = SPLITS_DIR / f"{split}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def extract_split(split, model, preprocess, device, batch_size):
    records = load_split(split)
    n = len(records)
    print(f"\n[{split}]  N={n}")

    basenames = [Path(r["resolved_image_path"]).name for r in records]
    labels = [r["emotion_label"] for r in records]

    feats = torch.empty((n, 512), dtype=torch.float32)

    t0 = time.time()
    for i in range(0, n, batch_size):
        batch_paths = [r["resolved_image_path"] for r in records[i : i + batch_size]]
        imgs = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        ).to(device)
        out = model.encode_image(imgs)
        feats[i : i + len(batch_paths)] = out.float().cpu()
    elapsed = time.time() - t0

    # Sanity checks
    nan_count = int(torch.isnan(feats).sum().item())
    norms = feats.norm(dim=1)
    print(
        f"  shape={tuple(feats.shape)}  dtype={feats.dtype}  "
        f"nan={nan_count}  "
        f"L2-norm mean={norms.mean().item():.3f}  std={norms.std().item():.3f}  "
        f"min={norms.min().item():.3f}  max={norms.max().item():.3f}"
    )
    print(f"  extract time: {elapsed:.1f}s  ({n / elapsed:.1f} img/s)")
    assert nan_count == 0, f"NaN in features for split={split}"
    assert feats.shape == (n, 512)

    out_path = FEATURES_DIR / f"clip_vitb32_{split}.pt"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"basenames": basenames, "features": feats, "labels": labels},
        out_path,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {out_path}  ({size_mb:.2f} MB)")
    return elapsed, size_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  batch_size={args.batch_size}  model={CLIP_MODEL}")

    model, preprocess = clip.load(CLIP_MODEL, device=device, jit=False)
    model.eval()

    total_time = 0.0
    total_mb = 0.0
    for split in SPLITS:
        t, mb = extract_split(split, model, preprocess, device, args.batch_size)
        total_time += t
        total_mb += mb

    print("\n" + "=" * 70)
    print(f"TOTAL extract time: {total_time:.1f}s    disk usage: {total_mb:.2f} MB")
    print(f"Outputs: {FEATURES_DIR}/")


if __name__ == "__main__":
    main()
