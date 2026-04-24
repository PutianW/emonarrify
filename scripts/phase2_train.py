"""
Phase 2 training script: CLIP image features -> MLP -> emotion embedding.

Preconditions:
    - data/features/clip_vitb32_{train,val}.pt produced by
      scripts/phase2_extract_clip.py
    - weights/emotion_lookup_table.json produced by Phase 3
      (will raise FileNotFoundError until Phase 3 delivers it — expected)

Loss terms (see plan Locked decisions #6–#9):
    L_cls  = CE( cos_sim(pred, lookup) / tau , label )
    L_mse  = mean((pred - target)^2) per sample, optionally class-weighted
    L      = { mse_only    -> L_mse
             { cls_only    -> L_cls
             { combined    -> L_mse + lambda_cls * L_cls    (default)

Model selection:
    Best checkpoint = epoch with highest 3-epoch moving average of
    val macro NN accuracy (trailing window of up to 3 recent epochs).
    Raw per-epoch macro is logged alongside but not used for selection.

Usage:
    python scripts/phase2_train.py --epochs 30 --lr 1e-3
    python scripts/phase2_train.py --loss-mode cls_only --seed 42
    python scripts/phase2_train.py --loss-reweight sqrt_inverse
"""
import argparse
import copy
import json
import os
import random
import sys
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from emonarrify.config import (
    EMOTION_LABELS,
    LOOKUP_TABLE_PATH,
    WEIGHTS_DIR,
    D_EMO,
)
from emonarrify.phase2.dataset import Phase2CachedDataset
from emonarrify.phase2.model import MLPProjectionHead, Phase2Model

NUM_CLASSES = len(EMOTION_LABELS)
CKPT_PATH = Path(WEIGHTS_DIR) / "phase2_mlp.pt"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 MLP training")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--loss-mode",
        choices=["mse_only", "cls_only", "combined"],
        default="combined",
    )
    p.add_argument(
        "--lambda-cls",
        type=float,
        default=0.5,
        help="Weight on L_cls. Silently ignored unless --loss-mode combined.",
    )
    p.add_argument(
        "--cls-temperature",
        type=float,
        default=0.07,
        help="Temperature tau for cosine-sim cross-entropy. CLIP/SimCLR default.",
    )
    p.add_argument(
        "--loss-reweight",
        choices=["none", "inverse", "sqrt_inverse"],
        default="none",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stop if moving-avg macro fails to improve for N epochs. 0 = off.",
    )
    return p.parse_args()


def load_lookup_matrix(device):
    """Load Phase 3's lookup table and stack into (NUM_CLASSES, D_EMO) in canonical order."""
    try:
        table = Phase2Model.load_lookup_table(LOOKUP_TABLE_PATH)
    except FileNotFoundError:
        print(
            f"\n[phase2_train] ERROR: lookup table not found at:\n  {LOOKUP_TABLE_PATH}\n"
            f"\nThis file is Phase 3's output (emotion_lookup_table.json) and is\n"
            f"required to compute regression targets for Phase 2 training.\n"
            f"\nPhase 2 code is ready; training will run end-to-end as soon as\n"
            f"the lookup table is delivered to that path.\n",
            file=sys.stderr,
        )
        sys.exit(2)

    missing = [lab for lab in EMOTION_LABELS if lab not in table]
    if missing:
        print(
            f"[phase2_train] ERROR: lookup table missing labels {missing}. "
            f"Expected all of {EMOTION_LABELS}.",
            file=sys.stderr,
        )
        sys.exit(2)

    rows = [table[lab] for lab in EMOTION_LABELS]
    return torch.stack(rows, dim=0).to(device)  # (NUM_CLASSES, D_EMO)


def compute_class_weights(train_labels, mode):
    """Return FloatTensor(NUM_CLASSES). sum(w) == NUM_CLASSES by construction when applicable."""
    if mode == "none":
        return torch.ones(NUM_CLASSES, dtype=torch.float32)

    counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for idx in train_labels:
        counts[int(idx)] += 1.0

    if (counts == 0).any():
        # Guard: no class should be empty in our train split, but fail loud if so.
        raise RuntimeError(f"empty class in train; counts={counts.tolist()}")

    if mode == "inverse":
        raw = 1.0 / counts
    elif mode == "sqrt_inverse":
        raw = 1.0 / counts.sqrt()
    else:
        raise ValueError(mode)

    # Normalize so sum(w) == NUM_CLASSES (keeps overall loss scale comparable).
    return raw * (NUM_CLASSES / raw.sum())


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
def compute_loss(
    pred,
    target_emb,
    labels,
    lookup_matrix,
    class_weights,
    loss_mode,
    lambda_cls,
    cls_temperature,
):
    """Returns (total_loss, L_mse_scalar_or_zero, L_cls_scalar_or_zero)."""
    zero = torch.zeros((), device=pred.device)
    l_mse = zero
    l_cls = zero

    if loss_mode in ("mse_only", "combined"):
        per_sample_mse = ((pred - target_emb) ** 2).mean(dim=1)  # (B,)
        sample_w = class_weights[labels]  # (B,)
        l_mse = (per_sample_mse * sample_w).mean()

    if loss_mode in ("cls_only", "combined"):
        # cos_sim: (B, NUM_CLASSES)
        pred_n = F.normalize(pred, dim=1)
        lookup_n = F.normalize(lookup_matrix, dim=1)
        logits = pred_n @ lookup_n.t() / cls_temperature
        l_cls = F.cross_entropy(logits, labels, weight=class_weights)

    if loss_mode == "mse_only":
        total = l_mse
    elif loss_mode == "cls_only":
        total = l_cls
    else:
        total = l_mse + lambda_cls * l_cls

    return total, l_mse.detach(), l_cls.detach()


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, lookup_matrix, device):
    """Run over loader and return a dict of metrics."""
    model.eval()
    all_pred = []
    all_target_emb = []
    all_labels = []

    for feats, labels, _basenames in loader:
        feats = feats.to(device)
        labels = labels.to(device)
        pred = model(feats)
        target_emb = lookup_matrix[labels]
        all_pred.append(pred)
        all_target_emb.append(target_emb)
        all_labels.append(labels)

    pred = torch.cat(all_pred, dim=0)
    target_emb = torch.cat(all_target_emb, dim=0)
    labels = torch.cat(all_labels, dim=0)

    mse = ((pred - target_emb) ** 2).mean().item()
    mean_cos_sim = F.cosine_similarity(pred, target_emb, dim=1).mean().item()

    pred_n = F.normalize(pred, dim=1)
    lookup_n = F.normalize(lookup_matrix, dim=1)
    cos_matrix = pred_n @ lookup_n.t()  # (N, NUM_CLASSES)
    preds_nn = cos_matrix.argmax(dim=1)

    overall_acc = (preds_nn == labels).float().mean().item()

    per_class_acc = []
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum().item() == 0:
            per_class_acc.append(float("nan"))
        else:
            per_class_acc.append((preds_nn[mask] == c).float().mean().item())

    valid = [a for a in per_class_acc if not (a != a)]  # drop NaN
    macro_acc = sum(valid) / len(valid) if valid else float("nan")

    return {
        "mse": mse,
        "cos_sim": mean_cos_sim,
        "overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
        "macro_acc": macro_acc,
        "preds_nn": preds_nn.cpu(),
        "labels": labels.cpu(),
    }


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, lookup_matrix, class_weights, args, device):
    model.train()
    total_mse = 0.0
    total_cls = 0.0
    total_loss = 0.0
    n_samples = 0

    for feats, labels, _basenames in loader:
        feats = feats.to(device)
        labels = labels.to(device)
        pred = model(feats)
        target_emb = lookup_matrix[labels]

        loss, l_mse, l_cls = compute_loss(
            pred, target_emb, labels, lookup_matrix, class_weights,
            args.loss_mode, args.lambda_cls, args.cls_temperature,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = feats.size(0)
        total_loss += loss.item() * bs
        total_mse += l_mse.item() * bs
        total_cls += l_cls.item() * bs
        n_samples += bs

    return {
        "loss": total_loss / n_samples,
        "mse": total_mse / n_samples,
        "cls": total_cls / n_samples,
    }


def fmt_pc(per_class):
    return "[" + " ".join(
        f"{lab}={ac:.2f}" if ac == ac else f"{lab}=NaN"
        for lab, ac in zip(EMOTION_LABELS, per_class)
    ) + "]"


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[config] {vars(args)}")
    print(f"[config] device={device}  ckpt={CKPT_PATH}")

    train_ds = Phase2CachedDataset("train")
    val_ds = Phase2CachedDataset("val")
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    lookup_matrix = load_lookup_matrix(device)  # sys.exit(2) if missing
    print(f"[lookup] shape={tuple(lookup_matrix.shape)}  order={EMOTION_LABELS}")

    class_weights = compute_class_weights(train_ds.label_idx.tolist(), args.loss_reweight).to(device)
    print(
        f"[reweight] mode={args.loss_reweight}  "
        f"weights={ {lab: round(float(class_weights[i]), 3) for i, lab in enumerate(EMOTION_LABELS)} }"
    )

    model = MLPProjectionHead(d_clip=512).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    macro_window = deque(maxlen=3)
    best_moving_avg = -float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, lookup_matrix, class_weights, args, device,
        )
        train_eval = evaluate(model, train_loader, lookup_matrix, device)
        val_eval = evaluate(model, val_loader, lookup_matrix, device)

        macro_window.append(val_eval["macro_acc"])
        moving_avg = sum(macro_window) / len(macro_window)

        print(
            f"[epoch {epoch:3d}] "
            f"train: loss={train_stats['loss']:.4f} mse={train_eval['mse']:.4f} "
            f"cos={train_eval['cos_sim']:.3f} acc={train_eval['overall_acc']:.3f}  "
            f"val: mse={val_eval['mse']:.4f} cos={val_eval['cos_sim']:.3f} "
            f"acc={val_eval['overall_acc']:.3f} "
            f"macro(raw)={val_eval['macro_acc']:.3f} "
            f"macro(ma3)={moving_avg:.3f}  "
            f"per-class={fmt_pc(val_eval['per_class_acc'])}"
        )

        improved = moving_avg > best_moving_avg
        if improved:
            best_moving_avg = moving_avg
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if args.patience > 0 and epochs_since_improve >= args.patience:
            print(
                f"[early-stop] moving-avg macro failed to improve for "
                f"{args.patience} epochs. Stopping at epoch {epoch}."
            )
            break

    if best_state is None:
        print("[warn] no checkpoint saved (training loop did not run)")
        return

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_moving_avg_macro": best_moving_avg,
            "args": vars(args),
            "emotion_labels": EMOTION_LABELS,
            "d_clip": 512,
            "d_emo": D_EMO,
        },
        CKPT_PATH,
    )
    print(
        f"[done] saved best checkpoint: epoch {best_epoch}, "
        f"moving-avg macro={best_moving_avg:.3f} -> {CKPT_PATH}"
    )


if __name__ == "__main__":
    main()
