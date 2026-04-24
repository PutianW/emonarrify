#!/usr/bin/env python3
"""
Adapt a pretrained VITS checkpoint from N_SRC speakers to N_DST speakers.

This script preserves all non-speaker-embedding weights (encoder, decoder, flow, etc.)
and adapts only the speaker embedding layer (emb_g) to the target speaker count.

Example:
    python adapt_speaker_embedding.py \
        --src-checkpoint vits/logs/emonarrify_phase3_vits_from_pretrained_emotion_only_clean/G_0.pth \
        --src-n-speakers 109 \
        --dst-n-speakers 10 \
        --output-checkpoint vits/logs/emonarrify_phase3_vits_lora_v2/G_0_adapted.pth \
        --init-method mean
"""

from __future__ import annotations

import argparse
import torch
from pathlib import Path
from typing import Literal


def adapt_checkpoint(
    src_ckpt_path: Path,
    src_n_speakers: int,
    dst_n_speakers: int,
    dst_ckpt_path: Path,
    init_method: Literal["mean", "random", "first"] = "mean",
) -> None:
    """Adapt speaker embedding in checkpoint from src to dst speaker count."""
    
    print(f"[Adapt] Loading source checkpoint: {src_ckpt_path}")
    src_ckpt = torch.load(src_ckpt_path, map_location="cpu")
    
    if "model" not in src_ckpt:
        raise ValueError("Checkpoint must have 'model' key (standard VITS format)")
    
    state_dict = src_ckpt["model"]
    
    # Check if speaker embedding exists
    if "emb_g.weight" not in state_dict:
        raise ValueError("Checkpoint does not contain 'emb_g.weight' (speaker embedding)")
    
    emb_g_src = state_dict["emb_g.weight"]
    assert emb_g_src.shape[0] == src_n_speakers, \
        f"Source embedding shape {emb_g_src.shape} doesn't match src_n_speakers={src_n_speakers}"
    
    embedding_dim = emb_g_src.shape[1]
    print(f"[Adapt] Source embedding: {emb_g_src.shape} (speakers={src_n_speakers}, dim={embedding_dim})")
    
    # Adapt speaker embedding
    if init_method == "mean":
        # Use mean of all source speakers as initialization for destination speakers
        mean_embedding = emb_g_src.mean(dim=0, keepdim=True)
        emb_g_dst = mean_embedding.expand(dst_n_speakers, embedding_dim).clone()
        print(f"[Adapt] Initializing {dst_n_speakers} speakers with mean of {src_n_speakers} source speakers")
    
    elif init_method == "random":
        # Random Gaussian initialization
        emb_g_dst = torch.randn(dst_n_speakers, embedding_dim) * emb_g_src.std()
        print(f"[Adapt] Initializing {dst_n_speakers} speakers with random Gaussian (σ={emb_g_src.std():.4f})")
    
    elif init_method == "first":
        # Pad/truncate to destination size
        if dst_n_speakers <= src_n_speakers:
            emb_g_dst = emb_g_src[:dst_n_speakers].clone()
            print(f"[Adapt] Using first {dst_n_speakers} speakers from source")
        else:
            # Pad with mean
            mean_embedding = emb_g_src.mean(dim=0, keepdim=True)
            pad_count = dst_n_speakers - src_n_speakers
            padding = mean_embedding.expand(pad_count, embedding_dim)
            emb_g_dst = torch.cat([emb_g_src, padding], dim=0)
            print(f"[Adapt] Using first {src_n_speakers} speakers + {pad_count} padded speakers")
    
    else:
        raise ValueError(f"Unknown init_method: {init_method}")
    
    # Replace speaker embedding
    state_dict["emb_g.weight"] = emb_g_dst
    print(f"[Adapt] Target embedding: {emb_g_dst.shape}")
    
    # Save adapted checkpoint with complete structure
    dst_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    adapted_ckpt = {
        "model": state_dict,
        "iteration": 0,  # Fresh start
        "learning_rate": 0.0001,  # Default learning rate
        "optimizer": None,  # No optimizer state in pretrained checkpoint
    }
    if "iteration" in src_ckpt:
        adapted_ckpt["iteration"] = src_ckpt["iteration"]
    if "learning_rate" in src_ckpt:
        adapted_ckpt["learning_rate"] = src_ckpt["learning_rate"]
    
    torch.save(adapted_ckpt, dst_ckpt_path)
    
    print(f"[Adapt] Saved adapted checkpoint to: {dst_ckpt_path}")
    print(f"[Adapt] ✓ Adaptation complete")


def main():
    parser = argparse.ArgumentParser(description="Adapt VITS speaker embedding checkpoint")
    parser.add_argument(
        "--src-checkpoint",
        type=str,
        required=True,
        help="Source VITS G_*.pth checkpoint path",
    )
    parser.add_argument(
        "--src-n-speakers",
        type=int,
        required=True,
        help="Number of speakers in source checkpoint",
    )
    parser.add_argument(
        "--dst-n-speakers",
        type=int,
        required=True,
        help="Target number of speakers",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=str,
        required=True,
        help="Output adapted checkpoint path",
    )
    parser.add_argument(
        "--init-method",
        type=str,
        default="mean",
        choices=["mean", "random", "first"],
        help="How to initialize speaker embeddings: mean=center of source, random=Gaussian, first=pad first N",
    )
    
    args = parser.parse_args()
    
    adapt_checkpoint(
        src_ckpt_path=Path(args.src_checkpoint),
        src_n_speakers=args.src_n_speakers,
        dst_n_speakers=args.dst_n_speakers,
        dst_ckpt_path=Path(args.output_checkpoint),
        init_method=args.init_method,
    )


if __name__ == "__main__":
    main()
