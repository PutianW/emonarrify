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
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_records(records: List[dict]) -> Tuple[List[dict], List[dict]]:
    train = [r for r in records if str(r.get("split", "")).lower() == "train"]
    val = [r for r in records if str(r.get("split", "")).lower() == "val"]
    return train, val


def _sanitize_text(text: str) -> str:
    # Keep one-line filelist entries stable for VITS parsers.
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def write_filelist(path: Path, records: List[dict], root_dir: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            wav = Path(str(r["audio_path"]))
            if not wav.is_absolute():
                wav = (root_dir / wav).resolve()
            text = _sanitize_text(r.get("narrative_text", ""))
            if not wav.exists() or not text:
                continue
            f.write(f"{wav}|{text}\n")
            written += 1
    return written


def patch_vits_config(
    src_config: Path,
    dst_config: Path,
    train_filelist: Path,
    val_filelist: Path,
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

    # Variant B: top-level fields
    cfg["training_files"] = cfg.get("training_files", train_str)
    cfg["validation_files"] = cfg.get("validation_files", val_str)
    cfg["training_files"] = train_str
    cfg["validation_files"] = val_str

    # Variant C: alternative names
    if "train_filelist_path" in cfg or "val_filelist_path" in cfg:
        cfg["train_filelist_path"] = train_str
        cfg["val_filelist_path"] = val_str

    dst_config.parent.mkdir(parents=True, exist_ok=True)
    with dst_config.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch VITS fine-tuning for EmoNarrify Phase3")
    parser.add_argument("--manifest", type=str, default="data/phase3_esd_manifest.jsonl")
    parser.add_argument("--vits-repo", type=str, required=True, help="Path to local VITS repo on AWS")
    parser.add_argument("--vits-config", type=str, required=True, help="Path to base VITS config JSON")
    parser.add_argument("--vits-train-script", type=str, default="train_ms.py", help="Training script inside VITS repo")
    parser.add_argument("--python-bin", type=str, default="python", help="Python executable in the active env")
    parser.add_argument("--run-name", type=str, default="emonarrify_phase3_vits")
    parser.add_argument("--work-dir", type=str, default="outputs/vits_phase3")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--no-launch", action="store_true", help="Only prepare filelists/config without starting training")
    parser.add_argument("--extra-args", type=str, default="", help="Extra args passed verbatim to VITS train script")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_dir = Path(__file__).resolve().parent
    manifest_path = Path(args.manifest).expanduser().resolve()
    vits_repo = Path(args.vits_repo).expanduser().resolve()
    vits_config = Path(args.vits_config).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run: PYTHONPATH=. python data/phase3_prepare_data.py --esd-root ESD --output data/phase3_esd_manifest.jsonl"
        )
    if not vits_repo.exists() or not vits_repo.is_dir():
        raise FileNotFoundError(f"VITS repo not found: {vits_repo}")
    if not vits_config.exists():
        raise FileNotFoundError(f"VITS config not found: {vits_config}")

    records = read_jsonl(manifest_path)
    train_records, val_records = split_records(records)
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

    n_train = write_filelist(train_filelist, train_records, root_dir=root_dir)
    n_val = write_filelist(val_filelist, val_records, root_dir=root_dir)

    if n_train == 0:
        raise RuntimeError("Generated train filelist is empty. Check audio paths in manifest.")

    generated_config = work_dir / "generated_vits_config.json"
    patch_vits_config(vits_config, generated_config, train_filelist, val_filelist)

    metadata = {
        "manifest": str(manifest_path),
        "vits_repo": str(vits_repo),
        "base_vits_config": str(vits_config),
        "generated_vits_config": str(generated_config),
        "train_filelist": str(train_filelist),
        "val_filelist": str(val_filelist),
        "num_train": n_train,
        "num_val": n_val,
        "run_name": args.run_name,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / "launcher_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[Phase3][VITS] Prepared train filelist: {train_filelist} ({n_train} entries)")
    print(f"[Phase3][VITS] Prepared val filelist:   {val_filelist} ({n_val} entries)")
    print(f"[Phase3][VITS] Generated config:       {generated_config}")

    train_script = (vits_repo / args.vits_train_script).resolve()
    if not train_script.exists():
        raise FileNotFoundError(
            f"VITS train script not found: {train_script}. "
            "Set --vits-train-script to your repo's actual training entry file."
        )

    cmd = [
        args.python_bin,
        str(train_script),
        "-c",
        str(generated_config),
        "-m",
        args.run_name,
    ]
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args.strip()))

    print("[Phase3][VITS] Launch command:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))

    if args.no_launch:
        print("[Phase3][VITS] --no-launch set; exiting after preparation.")
        return

    env = os.environ.copy()
    env["EMONARRIFY_VITS_TRAIN_FILELIST"] = str(train_filelist)
    env["EMONARRIFY_VITS_VAL_FILELIST"] = str(val_filelist)

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
