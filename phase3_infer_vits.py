"""
Short CLI for VITS inference on EmoNarrify Phase 3 runs.

Example:
    python phase3_infer_vits.py \
        --text "How are you doing" \
        --emotion happy \
        --speaker 0001 \
        --ckpt latest

Notes:
  - Current VITS path is text + speaker conditioning.
  - --emotion is accepted for UX compatibility, but is not used by current model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from scipy.io.wavfile import write as wav_write


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_vits_repo(project_root: Path, provided: Optional[str]) -> Path:
    if provided:
        p = Path(provided).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        return p
    return (project_root / "vits").resolve()


def _find_latest_checkpoint(model_dir: Path) -> Path:
    candidates = list(model_dir.glob("G_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No generator checkpoints found under: {model_dir}")

    def _iter_of(path: Path) -> int:
        stem = path.stem  # G_2000
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return -1

    candidates.sort(key=_iter_of)
    return candidates[-1]


def _load_speaker_map(metadata_path: Path) -> Dict[str, int]:
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    m = data.get("speaker_map")
    if isinstance(m, dict):
        return {str(k): int(v) for k, v in m.items()}
    return {}


def _load_emotion_map(metadata_path: Path) -> Dict[str, int]:
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    m = data.get("emotion_map")
    if isinstance(m, dict):
        return {str(k).strip().lower(): int(v) for k, v in m.items()}
    return {}


def _resolve_sid(raw_speaker: str, speaker_map: Dict[str, int], n_speakers: int) -> int:
    s = (raw_speaker or "").strip()
    if s in speaker_map:
        sid = speaker_map[s]
    else:
        try:
            sid = int(s)
        except ValueError as exc:
            valid_keys = ", ".join(sorted(speaker_map.keys())[:10])
            suffix = "..." if len(speaker_map) > 10 else ""
            raise ValueError(
                f"Invalid --speaker={raw_speaker!r}. Use a numeric sid or one of speaker ids: {valid_keys}{suffix}"
            ) from exc

    if sid < 0 or sid >= n_speakers:
        raise ValueError(f"Resolved speaker id {sid} out of range [0, {n_speakers - 1}]")
    return sid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short VITS inference CLI for EmoNarrify")
    parser.add_argument("--text", type=str, required=True, help="Input text to synthesize")
    parser.add_argument(
        "--emotion",
        type=str,
        default="neutral",
        help="Emotion label or numeric eid (used when config enables n_emotions > 0)",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="0001",
        help="Speaker key (e.g., 0001) or numeric sid",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="latest",
        help="Checkpoint path or 'latest'",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="emonarrify_phase3_vits_esd",
        help="VITS run directory name under vits/logs",
    )
    parser.add_argument(
        "--vits-repo",
        type=str,
        default=None,
        help="Path to VITS repository (default: ./vits)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="outputs/vits_phase3/launcher_metadata.json",
        help="Path to launcher metadata json (for speaker_map)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/vits_infer.wav",
        help="Output wav path",
    )
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = _resolve_project_root()

    vits_repo = _resolve_vits_repo(project_root, args.vits_repo)
    if not vits_repo.exists():
        raise FileNotFoundError(f"VITS repo not found: {vits_repo}")

    sys.path.insert(0, str(vits_repo))
    import commons  # pylint: disable=import-error
    import utils  # pylint: disable=import-error
    from models import SynthesizerTrn  # pylint: disable=import-error
    from text import text_to_sequence  # pylint: disable=import-error
    from text.symbols import symbols  # pylint: disable=import-error

    model_dir = vits_repo / "logs" / args.run_name
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")

    if args.ckpt.lower() == "latest":
        ckpt_path = _find_latest_checkpoint(model_dir)
    else:
        ckpt_path = Path(args.ckpt).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (project_root / ckpt_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    metadata_path = Path(args.metadata).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = (project_root / metadata_path).resolve()

    hps = utils.get_hparams_from_file(str(config_path))
    speaker_map = _load_speaker_map(metadata_path)
    emotion_map = _load_emotion_map(metadata_path)
    n_speakers = int(getattr(hps.data, "n_speakers", 0) or 0)
    sid = 0
    if n_speakers > 0:
        sid = _resolve_sid(args.speaker, speaker_map, n_speakers)
    n_emotions = int(getattr(hps.data, "n_emotions", 0) or 0)
    emotion_label = args.emotion.strip().lower()

    eid = 0
    if n_emotions > 0:
      if emotion_label in emotion_map:
          eid = int(emotion_map[emotion_label])
      else:
          try:
              eid = int(emotion_label)
          except ValueError as exc:
              valid = ", ".join(sorted(emotion_map.keys())) if emotion_map else "<no emotion_map found>"
              raise ValueError(
                  f"Emotion conditioning is enabled (n_emotions={n_emotions}) but --emotion={args.emotion!r} "
                  f"is not mappable. Valid labels: {valid} or provide numeric eid in [0, {n_emotions - 1}]"
              ) from exc
      if eid < 0 or eid >= n_emotions:
          raise ValueError(f"Resolved emotion id {eid} out of range [0, {n_emotions - 1}]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n_emotions <= 0 and args.emotion.strip().lower() != "neutral":
        print(
            f"[warn] --emotion {args.emotion!r} is currently NOT used by this VITS model; "
            "generation uses text + speaker only."
        )

    model_kwargs = dict(**hps.model)
    model_kwargs["n_emotions"] = n_emotions

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **model_kwargs,
    ).to(device)
    net_g.eval()

    _ = utils.load_checkpoint(str(ckpt_path), net_g, None)

    seq = text_to_sequence(args.text, hps.data.text_cleaners)
    if getattr(hps.data, "add_blank", False):
        seq = commons.intersperse(seq, 0)
    if not seq:
        raise ValueError("Text became empty after cleaners. Please provide more textual content.")

    x_tst = torch.LongTensor(seq).unsqueeze(0).to(device)
    x_tst_lengths = torch.LongTensor([x_tst.size(1)]).to(device)
    sid_t = torch.LongTensor([sid]).to(device)
    eid_t = torch.LongTensor([eid]).to(device)

    with torch.no_grad():
        audio = net_g.infer(
            x_tst,
            x_tst_lengths,
            sid=sid_t,
            eid=eid_t,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            length_scale=args.length_scale,
        )[0][0, 0].data.cpu().float().numpy()

    out_path = Path(args.output).expanduser()
    if not out_path.is_absolute():
        out_path = (project_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wav_write(str(out_path), int(hps.data.sampling_rate), audio)

    print("=" * 60)
    print(f"Saved:      {out_path}")
    print(f"Run:        {args.run_name}")
    print(f"Checkpoint: {ckpt_path}")
    if n_speakers > 0:
        print(f"Speaker:    {args.speaker} -> sid={sid}")
    else:
        print("Speaker:    disabled (n_speakers=0, emotion-only mode)")
    if n_emotions > 0:
        print(f"Emotion:    {args.emotion} -> eid={eid}")
    else:
        print(f"Emotion:    {args.emotion} (currently ignored)")
    print(f"SampleRate: {hps.data.sampling_rate}")
    print("=" * 60)


if __name__ == "__main__":
    main()
