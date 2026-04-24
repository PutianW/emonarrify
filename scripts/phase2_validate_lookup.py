"""
F1: Phase 3 lookup-table validation.

Read-only. Runs 6 structural checks against weights/emotion_lookup_table.json,
computes the 5x5 cosine-similarity matrix between emotion embeddings, reports
off-diagonal mean/max/min, and applies the three-tier geometric health rule:

    Healthy     : all off-diagonal < 0.3
    Borderline  : any off-diagonal in [0.3, 0.85] OR mean off-diagonal in [0.3, 0.85]
    Collapse    : mean off-diagonal > 0.85 OR any off-diagonal > 0.95

Exit codes:
    0 = healthy
    1 = borderline (user decision)
    2 = collapse (halt)
"""
import json
import math
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from emonarrify.config import EMOTION_LABELS, LOOKUP_TABLE_PATH, D_EMO
from emonarrify.phase2.model import Phase2Model


def _fail(msg):
    print(f"  [FAIL] {msg}")
    return False


def _pass(msg):
    print(f"  [ OK ] {msg}")
    return True


def main():
    path = LOOKUP_TABLE_PATH
    print(f"Validating lookup table at:\n  {path}\n")

    # Check 1: parse
    print("Check 1: json.load parses successfully")
    try:
        with open(path) as f:
            data = json.load(f)
        _pass("parsed OK")
    except Exception as e:
        _fail(f"parse error: {type(e).__name__}: {e}")
        sys.exit(2)

    # Check 2: embedding_dim == D_EMO
    print(f"\nCheck 2: data['embedding_dim'] == {D_EMO}")
    dim = data.get("embedding_dim")
    if dim == D_EMO:
        _pass(f"embedding_dim={dim}")
    else:
        _fail(f"embedding_dim={dim} (expected {D_EMO})")
        sys.exit(2)

    # Check 3: key set equals EMOTION_LABELS
    print(f"\nCheck 3: embeddings keys == set(EMOTION_LABELS)")
    embs = data.get("embeddings", {})
    got = set(embs.keys())
    expected = set(EMOTION_LABELS)
    if got == expected:
        _pass(f"keys match: {sorted(got)}")
    else:
        _fail(f"mismatch. missing={expected - got}  extra={got - expected}")
        sys.exit(2)

    # Check 4: each vector len == D_EMO, finite
    print(f"\nCheck 4: each vector len={D_EMO}, no NaN/Inf, numeric")
    all_ok = True
    for lab in EMOTION_LABELS:
        vec = embs[lab]
        if not isinstance(vec, list):
            all_ok &= _fail(f"{lab}: not a list (type={type(vec).__name__})")
            continue
        if len(vec) != D_EMO:
            all_ok &= _fail(f"{lab}: len={len(vec)} (expected {D_EMO})")
            continue
        if not all(isinstance(v, (int, float)) for v in vec):
            all_ok &= _fail(f"{lab}: non-numeric element found")
            continue
        if any(math.isnan(v) or math.isinf(v) for v in vec):
            all_ok &= _fail(f"{lab}: contains NaN or Inf")
            continue
        _pass(f"{lab}: len=256 finite")
    if not all_ok:
        sys.exit(2)

    # Check 5: 5x5 cosine similarity matrix
    print("\nCheck 5: 5x5 cosine similarity matrix (canonical order)")
    mat = torch.stack(
        [torch.tensor(embs[lab], dtype=torch.float64) for lab in EMOTION_LABELS],
        dim=0,
    )
    mat_n = F.normalize(mat, dim=1)
    cos_mat = mat_n @ mat_n.t()  # (5, 5)

    # Markdown table
    header = "| |" + "|".join(f" {lab} " for lab in EMOTION_LABELS) + "|"
    sep = "|---|" + "|".join("---" for _ in EMOTION_LABELS) + "|"
    print()
    print(header)
    print(sep)
    for i, rlab in enumerate(EMOTION_LABELS):
        row = f"| **{rlab}** |" + "|".join(f" {cos_mat[i, j].item():+.3f} " for j in range(5)) + "|"
        print(row)

    # Off-diagonal stats
    off_diag = []
    for i, j in combinations(range(5), 2):
        off_diag.append(cos_mat[i, j].item())
    mean_off = sum(off_diag) / len(off_diag)
    max_off = max(off_diag)
    min_off = min(off_diag)
    print(f"\n  off-diagonal pairs: {len(off_diag)}")
    print(f"  mean off-diagonal : {mean_off:+.4f}")
    print(f"  max  off-diagonal : {max_off:+.4f}")
    print(f"  min  off-diagonal : {min_off:+.4f}")

    # Check 6: load_lookup_table round-trip
    print("\nCheck 6: Phase2Model.load_lookup_table(...) round-trip")
    try:
        table = Phase2Model.load_lookup_table(path)
        assert set(table.keys()) == set(EMOTION_LABELS)
        for lab in EMOTION_LABELS:
            assert table[lab].shape == (D_EMO,)
            assert table[lab].dtype == torch.float32
        _pass("load_lookup_table returned dict with correct shapes/dtype")
    except Exception as e:
        _fail(f"load_lookup_table raised {type(e).__name__}: {e}")
        sys.exit(2)

    # Three-tier verdict
    print("\n" + "=" * 70)
    print("Geometric health verdict")
    print("=" * 70)

    any_gt_095 = max_off > 0.95
    mean_gt_085 = mean_off > 0.85
    any_in_borderline = any(0.3 <= v <= 0.85 for v in off_diag)
    mean_in_borderline = 0.3 <= mean_off <= 0.85
    all_lt_03 = max_off < 0.3

    if any_gt_095 or mean_gt_085:
        tier = "COLLAPSE"
        rc = 2
        print(
            f"  tier = {tier}  (mean_off={mean_off:.3f} > 0.85 ? {mean_gt_085}; "
            f"max_off={max_off:.3f} > 0.95 ? {any_gt_095})"
        )
        print("  -> halt Steps 3-5; escalate to user.")
    elif all_lt_03:
        tier = "HEALTHY"
        rc = 0
        print(
            f"  tier = {tier}  (all off-diagonal < 0.3; "
            f"max_off={max_off:.3f}, mean_off={mean_off:.3f})"
        )
        print("  -> safe to proceed to Step 3.")
    else:  # borderline
        tier = "BORDERLINE"
        rc = 1
        print(
            f"  tier = {tier}  "
            f"(any_in_[0.3,0.85]={any_in_borderline}, "
            f"mean_in_[0.3,0.85]={mean_in_borderline}, "
            f"max_off={max_off:.3f}, mean_off={mean_off:.3f})"
        )
        print("  -> report matrix + stats to user; await decision.")

    sys.exit(rc)


if __name__ == "__main__":
    main()
