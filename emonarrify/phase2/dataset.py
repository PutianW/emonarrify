"""
Phase 2 cached-feature Dataset.

Wraps the per-split CLIP feature cache produced by
scripts/phase2_extract_clip.py. Each item is a tuple:

    (feature: FloatTensor[512], label_idx: LongTensor[scalar], basename: str)

`basename` is passed through verbatim so downstream code can look up
misclassified samples, build confusion-matrix error galleries, and run
qualitative error analysis. `label_idx` is derived from EMOTION_TO_IDX.
"""
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..config import EMOTION_TO_IDX, PROJECT_ROOT

_VALID_SPLITS = ("train", "val", "test", "phase1_holdout")
_FEATURES_DIR = Path(PROJECT_ROOT) / "data" / "features"
_CACHE_TEMPLATE = "clip_vitb32_{split}.pt"


class Phase2CachedDataset(Dataset):
    """Load a pre-extracted CLIP feature cache for one split."""

    def __init__(self, split, features_dir=None):
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {_VALID_SPLITS}, got {split!r}"
            )
        cache_dir = Path(features_dir) if features_dir else _FEATURES_DIR
        cache_path = cache_dir / _CACHE_TEMPLATE.format(split=split)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"CLIP feature cache not found at {cache_path}. "
                f"Run scripts/phase2_extract_clip.py first."
            )

        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.features = blob["features"]
        self.basenames = blob["basenames"]
        self.labels = blob["labels"]

        if not (
            self.features.shape[0]
            == len(self.basenames)
            == len(self.labels)
        ):
            raise RuntimeError(
                f"cache length mismatch in {cache_path}: "
                f"features={self.features.shape[0]} "
                f"basenames={len(self.basenames)} "
                f"labels={len(self.labels)}"
            )

        unknown = [l for l in self.labels if l not in EMOTION_TO_IDX]
        if unknown:
            raise ValueError(
                f"unknown labels in {cache_path}: {set(unknown)}"
            )
        self.label_idx = torch.tensor(
            [EMOTION_TO_IDX[l] for l in self.labels], dtype=torch.long
        )

        self.split = split
        self.cache_path = cache_path

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.label_idx[idx], self.basenames[idx]


def _smoke_test():
    """Iterate each split once and report shapes. Run via `python -m emonarrify.phase2.dataset`."""
    from torch.utils.data import DataLoader

    for split in _VALID_SPLITS:
        ds = Phase2CachedDataset(split)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        print(f"[{split}] len={len(ds)}  cache={ds.cache_path}")

        seen = 0
        first_batch_reported = False
        for features, label_idx, basenames in loader:
            seen += features.shape[0]
            if not first_batch_reported:
                print(
                    f"  first batch: features={tuple(features.shape)} {features.dtype}  "
                    f"label_idx={tuple(label_idx.shape)} {label_idx.dtype}  "
                    f"basenames[0]={basenames[0]}  label_idx[0]={int(label_idx[0])}"
                )
                first_batch_reported = True
        print(f"  iterated {seen} samples across one epoch  -> OK\n")


if __name__ == "__main__":
    _smoke_test()
