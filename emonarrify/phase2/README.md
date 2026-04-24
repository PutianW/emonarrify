# Phase 2 — Image Emotion Encoder

Maps an RGB image to a 256-d emotion embedding in Phase 3's space:

    image -> frozen CLIP ViT-B/32 -> 512-d visual feature -> MLP -> 256-d emotion embedding

The MLP is the only trainable component. CLIP is frozen and its features are
pre-extracted to disk so training only sees cached tensors.

## Files

| path | role |
|---|---|
| `emonarrify/phase2/model.py` | `MLPProjectionHead`, `Phase2Model`, `Phase2Model.load_lookup_table` |
| `emonarrify/phase2/dataset.py` | `Phase2CachedDataset(split)` — wraps cached features |
| `data/phase2_build_splits.py` | stratified 70/15/15 on `phase1_vist_train_new.jsonl`; writes `data/splits/*.jsonl` |
| `scripts/phase2_extract_clip.py` | CLIP ViT-B/32 feature precompute → `data/features/clip_vitb32_*.pt` |
| `scripts/phase2_train.py` | MLP training (loss / reweight / macro-selection) |
| `scripts/phase2_eval.py` | test-set reports (confusion matrix, misclassified basenames) |

## Data dependencies

Training requires:

- `data/features/clip_vitb32_{train,val}.pt` — from `scripts/phase2_extract_clip.py`.
- `weights/emotion_lookup_table.json` — **Phase 3 output.** Until Phase 3
  delivers it, `scripts/phase2_train.py` exits with a clear message and
  nonzero status. All other steps (splits, CLIP extraction) run to completion.

`data/splits/phase1_holdout.jsonl` and `data/features/clip_vitb32_phase1_holdout.pt`
are reserved for future end-to-end pipeline evaluation. They are **not** used
in Phase 2 training, validation, or test.

## End-to-end pipeline (once Phase 3 delivers the lookup table)

```
python data/phase2_build_splits.py             # deterministic (seed=42)
python scripts/phase2_extract_clip.py          # CLIP features → data/features/
python scripts/phase2_train.py                 # trains MLP → weights/phase2_mlp.pt
python scripts/phase2_eval.py                  # reports on test split
```

## Baseline training

```
python scripts/phase2_train.py --loss-mode combined --lambda-cls 0.5
```

## Loss-composition ablation (3 runs, fixed seed for comparability)

```
python scripts/phase2_train.py --loss-mode mse_only  --seed 42
python scripts/phase2_train.py --loss-mode cls_only  --seed 42
python scripts/phase2_train.py --loss-mode combined  --lambda-cls 0.5 --seed 42
```

## Class-imbalance ablation

```
python scripts/phase2_train.py --loss-reweight none          --seed 42
python scripts/phase2_train.py --loss-reweight sqrt_inverse  --seed 42
python scripts/phase2_train.py --loss-reweight inverse       --seed 42
```

`--loss-reweight` and `--loss-mode` are orthogonal; reweighting applies to
whichever loss terms are active.

## Model selection

Best checkpoint = epoch with the highest **3-epoch moving average of val
macro NN accuracy** (trailing window of up to 3 recent epochs; epoch 1 = raw,
epoch 2 = mean of epochs 1–2, epoch N ≥ 3 = mean of epochs N−2 to N).

Raw per-epoch macro is logged but not used for selection (it jumps in ~33%
steps because the val "angry" class has only ~4 samples). Raw overall
accuracy is never used for selection (would pick a collapse-to-neutral
model, which scores ~60% overall but ~20% macro).

## CLI reference — `scripts/phase2_train.py`

| flag | default | meaning |
|---|---|---|
| `--epochs` | 30 | training epochs |
| `--lr` | 1e-3 | Adam learning rate |
| `--batch-size` | 64 | |
| `--weight-decay` | 1e-4 | Adam weight decay |
| `--device` | auto (cuda if available) | |
| `--seed` | 42 | |
| `--loss-mode` | `combined` | `mse_only` / `cls_only` / `combined` |
| `--lambda-cls` | 0.5 | only consumed when `--loss-mode combined`; silently ignored otherwise |
| `--cls-temperature` | 0.07 | temperature τ in `CE(cos_sim / τ, label)`. CLIP/SimCLR default |
| `--loss-reweight` | `none` | `none` / `inverse` / `sqrt_inverse` |
| `--patience` | 0 | early-stop N epochs without moving-avg improvement. 0 = disabled |
