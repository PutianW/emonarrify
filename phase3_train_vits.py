"""Phase 3 VITS fine-tuning launcher (AWS/tmux friendly).

This script does 3 things:
1) Read Phase3 manifest (JSONL)
2) Build VITS filelists (audio_path|text)
3) Launch VITS training script with a generated config

Example:
    PYTHONPATH=. python phase3_train_vits.py \
      --manifest data/phase3_esd_manifest.jsonl \
      --vits-repo ~/vits \
      --vits-config ~/vits/configs/finetune_emonarrify.json \
      --run-name emonarrify_phase3_vits
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import wave
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover - optional for envs without torch
    torch = None


# Global warning suppression (requested for cleaner training logs)
warnings.filterwarnings("ignore")


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_records(records: List[dict]) -> Tuple[List[dict], List[dict]]:
    train = [r for r in records if str(r.get("split", "")).lower() == "train"]
    val = [r for r in records if str(r.get("split", "")).lower() == "val"]
    return train, val


def _str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value!r}")


@dataclass
class FilelistStats:
    written: int
    skipped_missing: int
    skipped_empty_text: int


def _sanitize_text(text: str) -> str:
    # Keep one-line filelist entries stable for VITS parsers.
    return str(text).replace("\n", " ").replace("\r", " ").strip()


_VITS_ALLOWED_TEXT_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz;:,.!?¡¿—…\"«»“” ")


def sanitize_text_for_vits_symbols(text: str) -> str:
    """Filter text to VITS symbol inventory subset to avoid KeyError during indexing."""
    text = text.replace("-", " ").replace("'", " ")
    filtered = "".join(ch if ch in _VITS_ALLOWED_TEXT_CHARS else " " for ch in text)
    # collapse whitespace
    return " ".join(filtered.split())


def resolve_vits_repo(repo_arg: str, root_dir: Path) -> Path:
    """Resolve VITS repo with common fallback paths."""
    candidate = Path(repo_arg).expanduser()
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve()

    fallback_candidates = [
        Path.home() / "vits",
        root_dir / "vits",
    ]
    for fb in fallback_candidates:
        if fb.exists() and fb.is_dir():
            print(
                f"[Phase3][VITS][Warn] Provided --vits-repo not found: {candidate}. "
                f"Using fallback: {fb}"
            )
            return fb.resolve()

    raise FileNotFoundError(
        f"VITS repo not found: {candidate}. "
        f"Checked fallbacks: {', '.join(str(p) for p in fallback_candidates)}"
    )


def resolve_train_script(vits_repo: Path, requested: str) -> Path:
    """Resolve train script path, supporting auto mode."""
    req = str(requested).strip()
    if req and req.lower() != "auto":
        script = (vits_repo / req).resolve()
        if not script.exists():
            raise FileNotFoundError(
                f"VITS train script not found: {script}. "
                "Set --vits-train-script to your repo's actual training entry file."
            )
        return script

    # Auto mode: prefer multi-speaker first for ESD.
    for name in ("train_ms.py", "train.py"):
        script = (vits_repo / name).resolve()
        if script.exists():
            print(f"[Phase3][VITS] Auto-selected train script: {script.name}")
            return script

    raise FileNotFoundError(
        f"No train script found under {vits_repo}. Expected one of: train_ms.py, train.py"
    )


def print_gpu_training_hint(fp16_run: Optional[bool]) -> None:
    """Print lightweight tuning hints from available GPU memory."""
    if torch is None or not torch.cuda.is_available():
        print("[Phase3][VITS][Hint] CUDA not detected at launcher time. Use conservative batch_size in config.")
        return

    try:
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        name = props.name
    except Exception:
        return

    if total_gb < 12:
        batch_hint = 4
    elif total_gb < 20:
        batch_hint = 8
    elif total_gb < 32:
        batch_hint = 12
    else:
        batch_hint = 16

    fp16_hint = True if fp16_run is None else bool(fp16_run)
    print(
        f"[Phase3][VITS][Hint] GPU0={name} ({total_gb:.1f} GB). "
        f"Suggested start: batch_size≈{batch_hint}, fp16_run={fp16_hint}"
    )


def find_free_port(start: int = 29500, end: int = 29999) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free TCP port found in range [{start}, {end}]")


def build_speaker_map(records: List[dict], collapse_to_one: bool = False) -> Dict[str, int]:
    speaker_ids = sorted({str(r.get("speaker_id", "")).strip() for r in records if str(r.get("speaker_id", "")).strip()})
    if collapse_to_one:
        return {sid: 0 for sid in speaker_ids}
    return {sid: idx for idx, sid in enumerate(speaker_ids)}


def parse_speaker_filter(spec: str) -> Optional[set[str]]:
    raw = (spec or "").strip()
    if not raw:
        return None
    out: set[str] = set()
    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        if "-" in part:
            a, b = [x.strip() for x in part.split("-", 1)]
            if a.isdigit() and b.isdigit():
                start, end = int(a), int(b)
                if start > end:
                    start, end = end, start
                width = max(len(a), len(b), 4)
                for n in range(start, end + 1):
                    out.add(str(n).zfill(width))
                continue
        if part.isdigit() and len(part) < 4:
            out.add(str(int(part)).zfill(4))
        else:
            out.add(part)
    return out if out else None


def filter_records_by_speakers(records: List[dict], keep_speakers: Optional[set[str]]) -> List[dict]:
    if not keep_speakers:
        return records
    return [r for r in records if str(r.get("speaker_id", "")).strip() in keep_speakers]


def canonical_emotion_label(raw: str) -> str:
    v = str(raw or "").strip().lower()
    aliases = {
        "anger": "angry",
        "angry": "angry",
        "happiness": "happy",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "sadness": "sad",
        "surprise": "surprise",
        "surprised": "surprise",
    }
    return aliases.get(v, v)


def build_emotion_map(records: List[dict], preferred_order: Optional[List[str]] = None) -> Dict[str, int]:
    found = {canonical_emotion_label(r.get("emotion_label", "")) for r in records}
    found = {x for x in found if x}
    if preferred_order:
        ordered = [canonical_emotion_label(x) for x in preferred_order]
        ordered_unique = []
        for x in ordered:
            if x and x not in ordered_unique:
                ordered_unique.append(x)
        if found and not found.issubset(set(ordered_unique)):
            missing = sorted(found - set(ordered_unique))
            raise ValueError(f"Emotion labels found in data but missing in --emotion-labels: {missing}")
        labels = ordered_unique
    else:
        labels = sorted(found)
    if not labels:
        raise RuntimeError("No valid emotion_label found for emotion conditioning")
    return {emo: idx for idx, emo in enumerate(labels)}


def infer_sample_rate(records: List[dict], root_dir: Path, max_probe: int = 64) -> Optional[int]:
    """Infer dominant sample rate by probing up to max_probe wav files."""
    counts: Dict[int, int] = {}
    checked = 0
    for r in records:
        wav = Path(str(r.get("audio_path", "")))
        if not wav.is_absolute():
            wav = (root_dir / wav).resolve()
        if not wav.exists():
            continue
        try:
            with wave.open(str(wav), "rb") as wf:
                sr = int(wf.getframerate())
        except Exception:
            continue
        counts[sr] = counts.get(sr, 0) + 1
        checked += 1
        if checked >= max_probe:
            break
    if not counts:
        return None
    # Pick the most frequent inferred sample rate.
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def write_filelist(
    path: Path,
    records: List[dict],
    root_dir: Path,
    *,
    use_multispeaker: bool,
    speaker_to_idx: Optional[Dict[str, int]] = None,
    use_emotion_conditioning: bool = False,
    emotion_to_idx: Optional[Dict[str, int]] = None,
    sanitize_text: bool = True,
) -> FilelistStats:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_missing = 0
    skipped_empty_text = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            wav = Path(str(r["audio_path"]))
            if not wav.is_absolute():
                wav = (root_dir / wav).resolve()
            text = _sanitize_text(r.get("narrative_text", ""))
            if sanitize_text:
                text = sanitize_text_for_vits_symbols(text)
            if not wav.exists():
                skipped_missing += 1
                continue
            if not text:
                skipped_empty_text += 1
                continue

            if use_multispeaker:
                if speaker_to_idx is None:
                    raise ValueError("speaker_to_idx is required in multispeaker mode")
                raw_sid = str(r.get("speaker_id", "")).strip()
                if not raw_sid:
                    raise ValueError("Record is missing speaker_id required for multispeaker mode")
                if raw_sid not in speaker_to_idx:
                    raise ValueError(f"Unknown speaker_id {raw_sid!r} not found in speaker map")
                sid = speaker_to_idx[raw_sid]
                if use_emotion_conditioning:
                    if emotion_to_idx is None:
                        raise ValueError("emotion_to_idx is required when use_emotion_conditioning=True")
                    raw_emo = canonical_emotion_label(r.get("emotion_label", ""))
                    if raw_emo not in emotion_to_idx:
                        raise ValueError(f"Unknown emotion_label {raw_emo!r} not found in emotion map")
                    eid = emotion_to_idx[raw_emo]
                    f.write(f"{wav}|{sid}|{eid}|{text}\n")
                else:
                    f.write(f"{wav}|{sid}|{text}\n")
            else:
                f.write(f"{wav}|{text}\n")
            written += 1
    return FilelistStats(
        written=written,
        skipped_missing=skipped_missing,
        skipped_empty_text=skipped_empty_text,
    )


def patch_vits_config(
    src_config: Path,
    dst_config: Path,
    train_filelist: Path,
    val_filelist: Path,
    *,
    n_speakers: Optional[int] = None,
    n_emotions: Optional[int] = None,
    text_cleaners: Optional[List[str]] = None,
    cleaned_text: Optional[bool] = None,
    sampling_rate: Optional[int] = None,
    fp16_run: Optional[bool] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> Dict[str, object]:
    """Best-effort config patch for common VITS config variants."""
    with src_config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_str = str(train_filelist.resolve())
    val_str = str(val_filelist.resolve())

    # Variant A: nested data block
    if isinstance(cfg.get("data"), dict):
        cfg["data"]["training_files"] = train_str
        cfg["data"]["validation_files"] = val_str
        if n_speakers is not None:
            cfg["data"]["n_speakers"] = int(n_speakers)
        if n_emotions is not None:
            cfg["data"]["n_emotions"] = int(n_emotions)
        if text_cleaners is not None:
            cfg["data"]["text_cleaners"] = list(text_cleaners)
        if cleaned_text is not None:
            cfg["data"]["cleaned_text"] = bool(cleaned_text)
        if sampling_rate is not None:
            cfg["data"]["sampling_rate"] = int(sampling_rate)

    # Variant B: top-level fields
    cfg["training_files"] = cfg.get("training_files", train_str)
    cfg["validation_files"] = cfg.get("validation_files", val_str)
    cfg["training_files"] = train_str
    cfg["validation_files"] = val_str

    # Variant C: alternative names
    if "train_filelist_path" in cfg or "val_filelist_path" in cfg:
        cfg["train_filelist_path"] = train_str
        cfg["val_filelist_path"] = val_str

    if isinstance(cfg.get("train"), dict):
        if fp16_run is not None:
            cfg["train"]["fp16_run"] = bool(fp16_run)
        if epochs is not None:
            cfg["train"]["epochs"] = int(epochs)
        if batch_size is not None:
            cfg["train"]["batch_size"] = int(batch_size)
        if learning_rate is not None:
            cfg["train"]["learning_rate"] = float(learning_rate)

    if isinstance(cfg.get("model"), dict) and n_emotions is not None:
        cfg["model"]["n_emotions"] = int(n_emotions)

    dst_config.parent.mkdir(parents=True, exist_ok=True)
    with dst_config.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch VITS fine-tuning for EmoNarrify Phase3")
    parser.add_argument("--manifest", type=str, default="data/phase3_esd_manifest.jsonl")
    parser.add_argument(
        "--include-test",
        type=_str2bool,
        default=False,
        help="Include split=test records into training pool (useful when a target speaker exists only in test)",
    )
    parser.add_argument(
        "--speaker-filter",
        type=str,
        default="",
        help="Keep only selected speakers (e.g. '0011-0020' or '0011,0013,0018')",
    )
    parser.add_argument(
        "--collapse-speakers-to-one",
        type=_str2bool,
        default=False,
        help="Map all remaining speaker_ids to a single sid=0 (useful for pseudo single-speaker training with pooled data)",
    )
    parser.add_argument(
        "--emotion-conditioning",
        type=_str2bool,
        default=False,
        help="Enable explicit emotion conditioning with filelist format audio|sid|eid|text",
    )
    parser.add_argument(
        "--emotion-only",
        type=_str2bool,
        default=False,
        help="Train with emotion conditioning only (set n_speakers=0; speaker ids remain in filelist for compatibility)",
    )
    parser.add_argument(
        "--emotion-labels",
        type=str,
        default="angry,happy,neutral,sad,surprise",
        help="Ordered emotion labels for eid mapping when --emotion-conditioning=true",
    )
    parser.add_argument("--n-emotions", type=int, default=0, help="Override number of emotions in config (0=auto)")
    parser.add_argument("--vits-repo", type=str, required=True, help="Path to local VITS repo on AWS (fallbacks: ~/vits, <repo_root>/vits)")
    parser.add_argument("--vits-config", type=str, required=True, help="Path to base VITS config JSON")
    parser.add_argument("--vits-train-script", type=str, default="auto", help="Training script inside VITS repo (auto|train_ms.py|train.py|custom)")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable in the active env")
    parser.add_argument("--run-name", type=str, default="emonarrify_phase3_vits")
    parser.add_argument("--work-dir", type=str, default="outputs/vits_phase3")
    parser.add_argument("--stable-mode", action="store_true", help="Use conservative settings for long-running stability")
    parser.add_argument(
        "--master-port",
        type=int,
        default=0,
        help="DDP MASTER_PORT for VITS process group (0=auto-find free port)",
    )
    parser.add_argument(
        "--multispeaker",
        type=_str2bool,
        default=None,
        help="Force multispeaker mode true/false. Default: auto (true for train_ms.py, false otherwise)",
    )
    parser.add_argument("--n-speakers", type=int, default=0, help="Override number of speakers in config (0=auto in multispeaker)")
    parser.add_argument(
        "--text-cleaners",
        type=str,
        default="",
        help="Comma-separated override for data.text_cleaners, e.g. transliteration_cleaners",
    )
    parser.add_argument(
        "--cleaned-text",
        type=_str2bool,
        default=None,
        help="Override data.cleaned_text true/false",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=0,
        help="Override data.sampling_rate (0=auto-infer from training wav files)",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for VITS train/eval (0=auto)")
    parser.add_argument("--pin-memory", type=_str2bool, default=None, help="Override DataLoader pin_memory true/false")
    parser.add_argument("--fp16-run", type=_str2bool, default=None, help="Override config train.fp16_run")
    parser.add_argument("--epochs", type=int, default=0, help="Override config train.epochs (0=no override)")
    parser.add_argument("--batch-size", type=int, default=0, help="Override config train.batch_size (0=no override)")
    parser.add_argument("--learning-rate", type=float, default=0.0, help="Override config train.learning_rate (0=no override)")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--sanitize-text", type=_str2bool, default=False, help="Filter text to VITS-supported symbols before writing filelists")
    parser.add_argument("--resume", action="store_true", help="Resume in VITS log dir if checkpoint exists (default VITS behavior)")
    parser.add_argument("--no-launch", action="store_true", help="Only prepare filelists/config without starting training")
    parser.add_argument("--extra-args", type=str, default="", help="Extra args passed verbatim to VITS train script")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_dir = Path(__file__).resolve().parent
    manifest_candidate = Path(args.manifest).expanduser()
    if manifest_candidate.is_absolute():
        manifest_path = manifest_candidate.resolve()
    else:
        manifest_path = (root_dir / manifest_candidate).resolve()
    vits_repo = resolve_vits_repo(args.vits_repo, root_dir=root_dir)

    vits_config_candidate = Path(args.vits_config).expanduser()
    if vits_config_candidate.exists():
        vits_config = vits_config_candidate.resolve()
    else:
        # allow passing config relative to VITS repo (e.g. configs/vctk_base.json)
        vits_config = (vits_repo / vits_config_candidate).resolve()
    work_dir_candidate = Path(args.work_dir).expanduser()
    if work_dir_candidate.is_absolute():
        work_dir = work_dir_candidate.resolve()
    else:
        work_dir = (root_dir / work_dir_candidate).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run: PYTHONPATH=. python data/phase3_prepare_data.py --esd-root ESD --output data/phase3_esd_manifest.jsonl"
        )
    if not vits_config.exists():
        raise FileNotFoundError(f"VITS config not found: {vits_config}")

    train_script = resolve_train_script(vits_repo, args.vits_train_script)

    auto_multispeaker = "train_ms" in train_script.name
    use_multispeaker = auto_multispeaker if args.multispeaker is None else bool(args.multispeaker)
    if args.emotion_only and not args.emotion_conditioning:
        raise RuntimeError("--emotion-only requires --emotion-conditioning true")

    records = read_jsonl(manifest_path)
    train_records, val_records = split_records(records)
    test_records = [r for r in records if str(r.get("split", "")).lower() == "test"]
    if args.include_test and test_records:
        train_records = train_records + test_records
        print(f"[Phase3][VITS] include-test ON | added {len(test_records)} test records into training pool")

    keep_speakers = parse_speaker_filter(args.speaker_filter)
    if keep_speakers:
        train_records = filter_records_by_speakers(train_records, keep_speakers)
        val_records = filter_records_by_speakers(val_records, keep_speakers)
        print(f"[Phase3][VITS] Speaker filter ON | kept speakers: {sorted(keep_speakers)}")
        found_speakers = {str(r.get("speaker_id", "")).strip() for r in (train_records + val_records)}
        missing = sorted(set(keep_speakers) - found_speakers)
        if missing:
            print(f"[Phase3][VITS][Warn] Requested speakers not found in manifest/split: {missing}")

    if args.train_limit > 0:
        train_records = train_records[: args.train_limit]
    if args.val_limit > 0:
        val_records = val_records[: args.val_limit]

    if not train_records:
        raise RuntimeError("No train records in manifest. Check split labels.")
    if not val_records:
        print("[Phase3][VITS][Warn] No val records found; validation filelist will be empty.")

    filelist_dir = work_dir / "filelists"
    train_filelist = filelist_dir / "train.txt"
    val_filelist = filelist_dir / "val.txt"

    speaker_to_idx = None
    inferred_n_speakers = 0
    emotion_to_idx = None
    inferred_n_emotions = 0
    if use_multispeaker:
        speaker_to_idx = build_speaker_map(
            train_records + val_records,
            collapse_to_one=bool(args.collapse_speakers_to_one),
        )
        inferred_n_speakers = len(speaker_to_idx)
        if inferred_n_speakers == 0:
            raise RuntimeError("No valid speaker_id found in manifest for multispeaker mode")
        if args.collapse_speakers_to_one:
            inferred_n_speakers = 1
            print("[Phase3][VITS] Multispeaker mode ON | collapse-speakers-to-one=true | effective speakers: 1")
        else:
            print(f"[Phase3][VITS] Multispeaker mode ON | inferred speakers: {inferred_n_speakers}")
    else:
        print("[Phase3][VITS] Multispeaker mode OFF | writing audio|text filelists")

    if args.emotion_conditioning:
        if not use_multispeaker:
            raise RuntimeError("--emotion-conditioning currently requires multispeaker mode (train_ms.py)")
        preferred = [x.strip() for x in args.emotion_labels.split(",") if x.strip()]
        emotion_to_idx = build_emotion_map(train_records + val_records, preferred_order=preferred)
        inferred_n_emotions = len(emotion_to_idx)
        print(f"[Phase3][VITS] Emotion conditioning ON | inferred emotions: {inferred_n_emotions} | map: {emotion_to_idx}")

    train_stats = write_filelist(
        train_filelist,
        train_records,
        root_dir=root_dir,
        use_multispeaker=use_multispeaker,
        speaker_to_idx=speaker_to_idx,
        use_emotion_conditioning=bool(args.emotion_conditioning),
        emotion_to_idx=emotion_to_idx,
        sanitize_text=bool(args.sanitize_text),
    )
    val_stats = write_filelist(
        val_filelist,
        val_records,
        root_dir=root_dir,
        use_multispeaker=use_multispeaker,
        speaker_to_idx=speaker_to_idx,
        use_emotion_conditioning=bool(args.emotion_conditioning),
        emotion_to_idx=emotion_to_idx,
        sanitize_text=bool(args.sanitize_text),
    )

    if train_stats.written == 0:
        raise RuntimeError("Generated train filelist is empty. Check audio paths in manifest.")

    # Heuristic: VCTK English config fails on raw CJK chars if cleaned_text=True.
    any_non_ascii = any(any(ord(ch) > 127 for ch in str(r.get("narrative_text", ""))) for r in train_records[:2000])
    inferred_text_cleaners = None
    inferred_cleaned_text = None
    if any_non_ascii and not args.text_cleaners and args.cleaned_text is None:
        inferred_text_cleaners = ["transliteration_cleaners"]
        inferred_cleaned_text = False
        print(
            "[Phase3][VITS][Info] Detected non-ASCII training text; "
            "auto-setting data.text_cleaners=['transliteration_cleaners'], data.cleaned_text=false"
        )

    text_cleaners_override = None
    if args.text_cleaners.strip():
        text_cleaners_override = [x.strip() for x in args.text_cleaners.split(",") if x.strip()]
    elif inferred_text_cleaners is not None:
        text_cleaners_override = inferred_text_cleaners

    cleaned_text_override = args.cleaned_text if args.cleaned_text is not None else inferred_cleaned_text

    inferred_sr = infer_sample_rate(train_records, root_dir=root_dir, max_probe=64)
    sampling_rate_override: Optional[int]
    if args.sampling_rate > 0:
        sampling_rate_override = args.sampling_rate
    else:
        sampling_rate_override = inferred_sr
    if sampling_rate_override is not None:
        print(f"[Phase3][VITS] Using data.sampling_rate={sampling_rate_override}")

    # Stable-mode conservative defaults (can be overridden by explicit args)
    effective_fp16_run = args.fp16_run
    effective_num_workers = args.num_workers
    effective_pin_memory = args.pin_memory
    if args.stable_mode:
        if effective_fp16_run is None:
            effective_fp16_run = False
        if effective_num_workers <= 0:
            effective_num_workers = 2
        if effective_pin_memory is None:
            effective_pin_memory = False
        print(
            "[Phase3][VITS] Stable mode ON | "
            f"fp16_run={effective_fp16_run}, num_workers={effective_num_workers}, pin_memory={effective_pin_memory}"
        )
    else:
        if effective_num_workers <= 0:
            effective_num_workers = 4
        if effective_pin_memory is None:
            effective_pin_memory = torch.cuda.is_available() if torch is not None else False

    generated_config = work_dir / "generated_vits_config.json"
    requested_n_speakers = args.n_speakers if args.n_speakers > 0 else None
    final_n_speakers = requested_n_speakers
    if use_multispeaker and final_n_speakers is None:
        final_n_speakers = inferred_n_speakers
    if args.emotion_only:
        if final_n_speakers not in (None, 0):
            print(
                f"[Phase3][VITS][Info] emotion-only ON: overriding n_speakers {final_n_speakers} -> 0"
            )
        final_n_speakers = 0

    requested_n_emotions = args.n_emotions if args.n_emotions > 0 else None
    final_n_emotions = requested_n_emotions
    if args.emotion_conditioning and final_n_emotions is None:
        final_n_emotions = inferred_n_emotions

    cfg = patch_vits_config(
        vits_config,
        generated_config,
        train_filelist,
        val_filelist,
        n_speakers=final_n_speakers,
        n_emotions=final_n_emotions,
        text_cleaners=text_cleaners_override,
        cleaned_text=cleaned_text_override,
        sampling_rate=sampling_rate_override,
        fp16_run=effective_fp16_run,
        epochs=(args.epochs if args.epochs > 0 else None),
        batch_size=(args.batch_size if args.batch_size > 0 else None),
        learning_rate=(args.learning_rate if args.learning_rate > 0 else None),
    )

    print_gpu_training_hint(args.fp16_run)

    metadata = {
        "manifest": str(manifest_path),
        "vits_repo": str(vits_repo),
        "base_vits_config": str(vits_config),
        "generated_vits_config": str(generated_config),
        "train_filelist": str(train_filelist),
        "val_filelist": str(val_filelist),
        "num_train": train_stats.written,
        "num_val": val_stats.written,
        "skipped_train_missing_audio": train_stats.skipped_missing,
        "skipped_train_empty_text": train_stats.skipped_empty_text,
        "skipped_val_missing_audio": val_stats.skipped_missing,
        "skipped_val_empty_text": val_stats.skipped_empty_text,
        "multispeaker": use_multispeaker,
        "inferred_n_speakers": inferred_n_speakers,
        "configured_n_speakers": final_n_speakers,
        "speaker_map": speaker_to_idx,
        "collapse_speakers_to_one": bool(args.collapse_speakers_to_one),
        "emotion_conditioning": bool(args.emotion_conditioning),
        "emotion_only": bool(args.emotion_only),
        "inferred_n_emotions": inferred_n_emotions,
        "configured_n_emotions": final_n_emotions,
        "emotion_map": emotion_to_idx,
        "effective_text_cleaners": cfg.get("data", {}).get("text_cleaners") if isinstance(cfg.get("data"), dict) else None,
        "effective_cleaned_text": cfg.get("data", {}).get("cleaned_text") if isinstance(cfg.get("data"), dict) else None,
        "effective_sampling_rate": cfg.get("data", {}).get("sampling_rate") if isinstance(cfg.get("data"), dict) else None,
        "effective_n_emotions": cfg.get("data", {}).get("n_emotions") if isinstance(cfg.get("data"), dict) else None,
        "effective_n_speakers": cfg.get("data", {}).get("n_speakers") if isinstance(cfg.get("data"), dict) else None,
        "effective_fp16_run": cfg.get("train", {}).get("fp16_run") if isinstance(cfg.get("train"), dict) else None,
        "stable_mode": args.stable_mode,
        "effective_num_workers": effective_num_workers,
        "effective_pin_memory": effective_pin_memory,
        "run_name": args.run_name,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / "launcher_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[Phase3][VITS] Prepared train filelist: {train_filelist} ({train_stats.written} entries)")
    print(f"[Phase3][VITS] Prepared val filelist:   {val_filelist} ({val_stats.written} entries)")
    if train_stats.skipped_missing or train_stats.skipped_empty_text or val_stats.skipped_missing or val_stats.skipped_empty_text:
        print(
            "[Phase3][VITS] Skipped records "
            f"train(missing={train_stats.skipped_missing}, empty_text={train_stats.skipped_empty_text}) "
            f"val(missing={val_stats.skipped_missing}, empty_text={val_stats.skipped_empty_text})"
        )
    print(f"[Phase3][VITS] Generated config:       {generated_config}")

    cmd = [
        args.python_bin,
        str(train_script),
        "-c",
        str(generated_config),
        "-m",
        args.run_name,
    ]
    if args.resume:
        print("[Phase3][VITS] Resume requested: VITS will auto-load latest checkpoint from logs/<run-name>/ if available.")
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args.strip()))

    print("[Phase3][VITS] Launch command:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))

    if args.no_launch:
        print("[Phase3][VITS] --no-launch set; exiting after preparation.")
        return

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    env["NCCL_DEBUG"] = "ERROR"
    env["EMONARRIFY_VITS_NUM_WORKERS"] = str(effective_num_workers)
    env["EMONARRIFY_VITS_PIN_MEMORY"] = "1" if effective_pin_memory else "0"
    selected_port = args.master_port if args.master_port > 0 else find_free_port()
    env["MASTER_ADDR"] = env.get("MASTER_ADDR", "localhost")
    env["MASTER_PORT"] = str(selected_port)
    env["EMONARRIFY_VITS_TRAIN_FILELIST"] = str(train_filelist)
    env["EMONARRIFY_VITS_VAL_FILELIST"] = str(val_filelist)
    print(f"[Phase3][VITS] Using DDP endpoint: {env['MASTER_ADDR']}:{env['MASTER_PORT']}")

    metadata["master_addr"] = env["MASTER_ADDR"]
    metadata["master_port"] = selected_port
    with (work_dir / "launcher_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log_path = work_dir / "vits_train.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n=== Launch ===\n")
        log_file.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(vits_repo),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)

        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"VITS training exited with code {ret}. Check log: {log_path}")

    print(f"[Phase3][VITS] Training finished. Log: {log_path}")


if __name__ == "__main__":
    main()
