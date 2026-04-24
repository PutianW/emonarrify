"""
Phase 2 split builder.

Reads phase1_vist_train_new.jsonl and phase1_vist_test_new.jsonl, translates
each jsonl image_path ("/emonarrify/images/..." -> "/emonarrify/data/images/..."),
verifies the resolved file exists, then writes:

  data/splits/train.jsonl           stratified 70% of train_new
  data/splits/val.jsonl             stratified 15% of train_new
  data/splits/test.jsonl            stratified 15% of train_new
  data/splits/phase1_holdout.jsonl  all of test_new (Phase 1 LoRA holdout,
                                    reserved for future end-to-end eval;
                                    NOT used in Phase 2 train/val/test)

Each output record carries the original fields plus a `resolved_image_path`
(absolute, existence-verified). Records whose resolved path is missing are
dropped and logged.

Deterministic: random_state=42 via sklearn.model_selection.train_test_split.
Re-running the script on the same inputs produces byte-identical outputs.
"""
import json
import os
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent
SPLITS_DIR = DATA_DIR / "splits"

EMOTION_LABELS = ["neutral", "happy", "angry", "sad", "surprise"]
SEED = 42

OLD_PATH_PREFIX = "/emonarrify/images/"
NEW_PATH_PREFIX = "/emonarrify/data/images/"


def translate(p):
    return p.replace(OLD_PATH_PREFIX, NEW_PATH_PREFIX)


def load_and_resolve(jsonl_path):
    kept, dropped = [], []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            resolved = translate(rec["image_path"])
            if os.path.isfile(resolved):
                rec["resolved_image_path"] = resolved
                kept.append(rec)
            else:
                dropped.append(resolved)
    return kept, dropped


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report_dist(name, records):
    c = Counter(r["emotion_label"] for r in records)
    total = len(records)
    print(f"\n  {name}  (N={total})")
    for lab in EMOTION_LABELS:
        n = c.get(lab, 0)
        pct = 100.0 * n / total if total else 0.0
        print(f"    {lab:10s}  {n:5d}  ({pct:5.1f}%)")
    extra = set(c) - set(EMOTION_LABELS)
    if extra:
        print(f"    WARNING unknown labels: { {k: c[k] for k in extra} }")


def main():
    train_new_path = DATA_DIR / "phase1_vist_train_new.jsonl"
    test_new_path = DATA_DIR / "phase1_vist_test_new.jsonl"

    print(f"Loading {train_new_path.name}")
    train_new, dropped_tr = load_and_resolve(train_new_path)
    print(f"  kept={len(train_new)}  dropped(missing file)={len(dropped_tr)}")

    print(f"Loading {test_new_path.name}")
    test_new, dropped_te = load_and_resolve(test_new_path)
    print(f"  kept={len(test_new)}  dropped(missing file)={len(dropped_te)}")

    # Stratified 70/15/15 on train_new only. Two-step: 70% / 30%, then split the 30% 50/50.
    labels = [r["emotion_label"] for r in train_new]
    train_recs, temp_recs, _, y_temp = train_test_split(
        train_new, labels,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
        shuffle=True,
    )
    val_recs, test_recs, _, _ = train_test_split(
        temp_recs, y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
        shuffle=True,
    )

    write_jsonl(SPLITS_DIR / "train.jsonl", train_recs)
    write_jsonl(SPLITS_DIR / "val.jsonl", val_recs)
    write_jsonl(SPLITS_DIR / "test.jsonl", test_recs)
    write_jsonl(SPLITS_DIR / "phase1_holdout.jsonl", test_new)

    print("\n" + "=" * 70)
    print("Per-split class distribution")
    print("=" * 70)
    report_dist("train.jsonl", train_recs)
    report_dist("val.jsonl", val_recs)
    report_dist("test.jsonl", test_recs)
    report_dist(
        "phase1_holdout.jsonl  [Phase 1 LoRA holdout; NOT used in Phase 2 train/val/test]",
        test_new,
    )

    print()
    print(
        f"Totals: train={len(train_recs)}  val={len(val_recs)}  "
        f"test={len(test_recs)}  phase1_holdout={len(test_new)}"
    )
    print(
        f"Dropped (missing resolved file): "
        f"train_new={len(dropped_tr)}  test_new={len(dropped_te)}"
    )
    print(f"Outputs written to {SPLITS_DIR}/")


if __name__ == "__main__":
    main()
