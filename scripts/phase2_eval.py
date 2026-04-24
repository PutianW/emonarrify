"""
Phase 2 evaluation script.

Loads weights/phase2_mlp.pt, runs inference on the test split's cached CLIP
features, and reports:
    - overall + per-class mean cosine similarity to the true-class lookup embedding
    - overall NN accuracy (argmax cos_sim over 5 lookup embeddings)
    - per-class + macro NN accuracy
    - 5x5 confusion matrix
    - misclassified sample basenames grouped by (true, predicted) label pair

Preconditions:
    - weights/phase2_mlp.pt produced by scripts/phase2_train.py
    - data/features/clip_vitb32_test.pt produced by scripts/phase2_extract_clip.py
    - weights/emotion_lookup_table.json (Phase 3 output)

Usage:
    python scripts/phase2_eval.py
    python scripts/phase2_eval.py --split val --weights weights/phase2_mlp.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from emonarrify.config import EMOTION_LABELS, LOOKUP_TABLE_PATH, WEIGHTS_DIR
from emonarrify.phase2.dataset import Phase2CachedDataset
from emonarrify.phase2.model import MLPProjectionHead, Phase2Model

NUM_CLASSES = len(EMOTION_LABELS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights",
        type=str,
        default=str(Path(WEIGHTS_DIR) / "phase2_mlp.pt"),
    )
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test", "phase1_holdout"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-misclassified-per-cell", type=int, default=10,
                   help="Cap basenames shown per (true,pred) cell in error table.")
    return p.parse_args()


def load_lookup_matrix(device):
    try:
        table = Phase2Model.load_lookup_table(LOOKUP_TABLE_PATH)
    except FileNotFoundError:
        print(
            f"\n[phase2_eval] ERROR: lookup table not found at:\n  {LOOKUP_TABLE_PATH}\n"
            f"Required for evaluation (to compute targets and NN similarity).\n",
            file=sys.stderr,
        )
        sys.exit(2)
    rows = [table[lab] for lab in EMOTION_LABELS]
    return torch.stack(rows, dim=0).to(device)


def load_checkpoint(path, device):
    if not Path(path).is_file():
        print(f"\n[phase2_eval] ERROR: weights not found at {path}. "
              f"Run scripts/phase2_train.py first.\n", file=sys.stderr)
        sys.exit(2)
    blob = torch.load(path, map_location=device, weights_only=False)
    d_clip = blob.get("d_clip", 512)
    model = MLPProjectionHead(d_clip=d_clip).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


@torch.no_grad()
def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[config] split={args.split}  weights={args.weights}  device={device}")

    lookup_matrix = load_lookup_matrix(device)
    model, blob = load_checkpoint(args.weights, device)
    print(
        f"[checkpoint] best_epoch={blob.get('best_epoch')}  "
        f"best_moving_avg_macro={blob.get('best_moving_avg_macro'):.3f}  "
        f"trained_with={blob.get('args')}"
    )

    ds = Phase2CachedDataset(args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"[data] {args.split}  N={len(ds)}")

    all_pred, all_labels, all_basenames = [], [], []
    for feats, labels, basenames in loader:
        feats = feats.to(device)
        labels = labels.to(device)
        pred = model(feats)
        all_pred.append(pred)
        all_labels.append(labels)
        all_basenames.extend(basenames)

    pred = torch.cat(all_pred, dim=0)
    labels = torch.cat(all_labels, dim=0)
    target_emb = lookup_matrix[labels]

    # Cosine sim to true-class embedding
    cos_true = F.cosine_similarity(pred, target_emb, dim=1)
    overall_cos = cos_true.mean().item()

    # NN accuracy
    pred_n = F.normalize(pred, dim=1)
    lookup_n = F.normalize(lookup_matrix, dim=1)
    cos_matrix = pred_n @ lookup_n.t()
    preds_nn = cos_matrix.argmax(dim=1)

    overall_acc = (preds_nn == labels).float().mean().item()

    per_class_cos = []
    per_class_acc = []
    per_class_n = []
    for c in range(NUM_CLASSES):
        mask = labels == c
        n_c = int(mask.sum().item())
        per_class_n.append(n_c)
        if n_c == 0:
            per_class_cos.append(float("nan"))
            per_class_acc.append(float("nan"))
        else:
            per_class_cos.append(cos_true[mask].mean().item())
            per_class_acc.append((preds_nn[mask] == c).float().mean().item())

    valid_accs = [a for a in per_class_acc if a == a]
    macro_acc = sum(valid_accs) / len(valid_accs) if valid_accs else float("nan")

    # Confusion matrix (rows=true, cols=pred)
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    for t, p in zip(labels.tolist(), preds_nn.tolist()):
        conf[t, p] += 1

    # --- report ---
    print("\n" + "=" * 72)
    print(f"Overall mean cosine sim : {overall_cos:.4f}")
    print(f"Overall NN accuracy     : {overall_acc:.4f}")
    print(f"Macro NN accuracy       : {macro_acc:.4f}  "
          f"(mean of per-class accuracies; excludes empty classes)")

    print("\nPer-class breakdown:")
    hdr = f"  {'label':<10} {'N':>5} {'cos_sim':>10} {'acc':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for lab, n_c, c_cos, c_acc in zip(
        EMOTION_LABELS, per_class_n, per_class_cos, per_class_acc
    ):
        cos_s = f"{c_cos:.4f}" if c_cos == c_cos else "  NaN  "
        acc_s = f"{c_acc:.4f}" if c_acc == c_acc else "  NaN  "
        print(f"  {lab:<10} {n_c:>5} {cos_s:>10} {acc_s:>10}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    hdr = " " * 12 + "".join(f"{lab:>10}" for lab in EMOTION_LABELS)
    print(hdr)
    for i, true_lab in enumerate(EMOTION_LABELS):
        row = "".join(f"{int(conf[i, j]):>10}" for j in range(NUM_CLASSES))
        print(f"  {true_lab:<10}" + row)

    # Misclassified samples
    print("\nMisclassified samples by (true, predicted) cell:")
    mis = [
        (int(t), int(p), bn)
        for t, p, bn in zip(labels.tolist(), preds_nn.tolist(), all_basenames)
        if t != p
    ]
    if not mis:
        print("  (none)")
    else:
        by_cell = {}
        for t, p, bn in mis:
            by_cell.setdefault((t, p), []).append(bn)
        for (t, p), basenames in sorted(by_cell.items()):
            cap = args.max_misclassified_per_cell
            head = basenames[:cap]
            more = len(basenames) - len(head)
            extra = f" (+{more} more)" if more > 0 else ""
            print(
                f"  {EMOTION_LABELS[t]} -> {EMOTION_LABELS[p]}  "
                f"(n={len(basenames)}): {head}{extra}"
            )


if __name__ == "__main__":
    main()
