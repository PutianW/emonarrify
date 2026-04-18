"""Phase 3 training script (CLI).

Run (example):
    PYTHONPATH=. python phase3_train.py \
      --manifest data/phase3_esd_manifest.jsonl \
      --epochs 5 \
      --batch-size 8 \
      --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from emonarrify.config import EMOTION_LABELS, LOOKUP_TABLE_PATH, SAMPLE_RATE
from emonarrify.phase3.model import ConditionalNeuralTTSBackbone, EmotionLookupTable


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resample_waveform_np(wav: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return wav.astype(np.float32)
    wav_t = torch.tensor(wav, dtype=torch.float32)
    if wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0).unsqueeze(0)
    else:
        wav_t = wav_t.mean(dim=-1, keepdim=False).unsqueeze(0).unsqueeze(0)
    new_len = max(1, int(round(wav_t.shape[-1] * tgt_sr / src_sr)))
    out = F.interpolate(wav_t, size=new_len, mode="linear", align_corners=False)
    return out.squeeze().cpu().numpy().astype(np.float32)


class ESDPhase3Dataset(Dataset):
    def __init__(self, items: List[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _emotion_to_idx(label: str) -> int:
        return EMOTION_LABELS.index(str(label).strip().lower())

    def __getitem__(self, idx: int) -> dict:
        rec = self.items[idx]
        text = rec["narrative_text"]
        emotion_label = str(rec["emotion_label"]).strip().lower()
        emotion_idx = self._emotion_to_idx(emotion_label)

        audio_path = Path(rec["audio_path"])
        wav, sr = sf.read(str(audio_path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = resample_waveform_np(wav, sr, SAMPLE_RATE)

        return {
            "text": text,
            "emotion_label": emotion_label,
            "emotion_idx": emotion_idx,
            "gt_wav": torch.tensor(wav, dtype=torch.float32),
            "audio_path": str(audio_path),
        }


def collate_fn(batch: List[dict]) -> List[dict]:
    return batch


def align_pred_to_target(pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
    if pred_wav.dim() == 1:
        pred_wav = pred_wav.unsqueeze(0).unsqueeze(0)
    else:
        pred_wav = pred_wav.unsqueeze(0)
    target_len = target_wav.shape[-1]
    pred_aligned = F.interpolate(pred_wav, size=target_len, mode="linear", align_corners=False)
    return pred_aligned.squeeze(0).squeeze(0)


def stft_mag_loss(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 512, hop_length: int = 128) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=pred.device)
    pred_stft = torch.stft(pred, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, return_complex=True)
    tgt_stft = torch.stft(target, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, return_complex=True)
    return F.l1_loss(torch.abs(pred_stft), torch.abs(tgt_stft))


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase3 emotion-conditioned TTS (CLI)")
    parser.add_argument("--manifest", type=str, default="data/phase3_esd_manifest.jsonl")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--spec-loss-weight", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=0, help="Use only first N train samples (0 = all)")
    parser.add_argument("--val-limit", type=int, default=0, help="Use only first N val samples (0 = all)")
    parser.add_argument("--save-dir", type=str, default="weights")
    parser.add_argument("--checkpoint-name", type=str, default="phase3_tts.pt")
    parser.add_argument("--lookup-out", type=str, default=LOOKUP_TABLE_PATH)
    parser.add_argument("--metrics-out", type=str, default="outputs/phase3_train_metrics.json")
    parser.add_argument("--save-val-audio", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase3][Train] Device: {device}")

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run: PYTHONPATH=. python data/phase3_prepare_data.py --esd-root ESD --output data/phase3_esd_manifest.jsonl"
        )

    records = read_jsonl(manifest_path)
    train_records = [r for r in records if str(r.get("split", "")).lower() == "train"]
    val_records = [r for r in records if str(r.get("split", "")).lower() == "val"]

    if args.train_limit > 0:
        train_records = train_records[: args.train_limit]
    if args.val_limit > 0:
        val_records = val_records[: args.val_limit]

    if not train_records:
        raise RuntimeError("No training samples found. Check manifest split labels.")
    if not val_records:
        print("[Phase3][Train][Warn] No validation samples found. val_loss will be NaN.")

    train_loader = DataLoader(
        ESDPhase3Dataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ESDPhase3Dataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"[Phase3][Train] records={len(records)} "
        f"train={len(train_records)} val={len(val_records)} batch={args.batch_size}"
    )

    lookup = EmotionLookupTable().to(device)
    tts_backbone = ConditionalNeuralTTSBackbone(sample_rate=SAMPLE_RATE).to(device)

    optimizer = torch.optim.AdamW(
        list(tts_backbone.parameters()) + list(lookup.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=min(5e-6, args.lr * 0.1),
    )

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / args.checkpoint_name
    best_ckpt_path = save_dir / f"best_{args.checkpoint_name}"

    def batch_forward_and_loss(batch: List[dict]) -> torch.Tensor:
        total_loss = 0.0
        for sample in batch:
            text = sample["text"]
            gt = sample["gt_wav"].to(device)
            emo_idx = torch.tensor(sample["emotion_idx"], device=device, dtype=torch.long)

            emo_emb = lookup(emo_idx)
            pred = tts_backbone.synthesize(text=text, emotion_embedding=emo_emb)
            pred = align_pred_to_target(pred, gt)

            loss_l1 = F.l1_loss(pred, gt)
            loss_spec = stft_mag_loss(pred, gt)
            loss = loss_l1 + args.spec_loss_weight * loss_spec
            total_loss = total_loss + loss
        return total_loss / max(1, len(batch))

    best_val = float("inf")
    history: List[EpochMetrics] = []

    for epoch in range(1, args.epochs + 1):
        tts_backbone.train()
        lookup.train()

        train_losses = []
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss = batch_forward_and_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(tts_backbone.parameters()) + list(lookup.parameters()),
                max_norm=args.grad_clip,
            )
            optimizer.step()

            train_losses.append(loss.item())
            if args.log_every > 0 and step % args.log_every == 0:
                print(
                    f"[Epoch {epoch}] step {step}/{len(train_loader)} "
                    f"train_loss={np.mean(train_losses[-args.log_every:]):.4f}"
                )

        tts_backbone.eval()
        lookup.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_loss = batch_forward_and_loss(batch)
                val_losses.append(val_loss.item())

        epoch_train = float(np.mean(train_losses)) if train_losses else float("nan")
        epoch_val = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append(EpochMetrics(epoch=epoch, train_loss=epoch_train, val_loss=epoch_val))

        print(f"[Epoch {epoch}] train={epoch_train:.4f} | val={epoch_val:.4f}")

        state = {
            "tts_model": tts_backbone.state_dict(),
            "lookup_table": lookup.state_dict(),
            "epoch": epoch,
            "train_loss": epoch_train,
            "val_loss": epoch_val,
            "sample_rate": SAMPLE_RATE,
            "model_name": "ConditionalNeuralTTSBackbone",
            "args": vars(args),
        }
        torch.save(state, ckpt_path)

        if not math.isnan(epoch_val) and epoch_val < best_val:
            best_val = epoch_val
            torch.save(state, best_ckpt_path)
            print(f"  -> saved best checkpoint: {best_ckpt_path}")

        scheduler.step()

    lookup.export_json(args.lookup_out)

    metrics_out = Path(args.metrics_out).expanduser().resolve()
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": str(manifest_path),
                "num_train": len(train_records),
                "num_val": len(val_records),
                "best_val_loss": best_val,
                "history": [asdict(m) for m in history],
                "checkpoint_path": str(ckpt_path),
                "best_checkpoint_path": str(best_ckpt_path),
                "lookup_json_path": str(Path(args.lookup_out).resolve()),
            },
            f,
            indent=2,
        )
    print(f"[Phase3][Train] Metrics saved: {metrics_out}")

    if args.save_val_audio:
        out_dir = Path("outputs").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        val_text = "A quiet scene unfolds under a pale sky, carrying a sense of calm solitude."
        lookup.eval()
        tts_backbone.eval()
        for emo in EMOTION_LABELS:
            emo_idx = torch.tensor(EMOTION_LABELS.index(emo), device=device, dtype=torch.long)
            with torch.no_grad():
                emo_emb = lookup(emo_idx)
                wav = tts_backbone.synthesize(val_text, emo_emb).detach().cpu().numpy().astype(np.float32)
            out_path = out_dir / f"phase3_val_{emo}.wav"
            sf.write(str(out_path), wav, SAMPLE_RATE)
            print(f"[Phase3][Train] Saved validation sample: {out_path}")

    print("[Phase3][Train] Done.")


if __name__ == "__main__":
    main()
