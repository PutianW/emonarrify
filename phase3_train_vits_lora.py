"""LoRA-style Phase3 VITS launcher.

This script keeps the existing Phase3 data/filelist pipeline, then injects
LoRA+PEFT settings into the generated VITS config before launching training.

Key idea:
- Keep speaker conditioning ON (do NOT use emotion-only)
- Add emotion conditioning + low-rank emotion adapter
- Optionally train only LoRA + emotion lookup table to preserve base speaking quality
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def _str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value!r}")


def find_free_port(start: int = 29500, end: int = 29999) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free TCP port in [{start}, {end}]")


def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def _bootstrap_pretrained_if_needed(args: argparse.Namespace, root: Path, vits_repo: Path) -> None:
    if not args.bootstrap_pretrained:
        print("[LoRA] bootstrap_pretrained=false; skip copying G_0/D_0")
        return

    dst_run_dir = (vits_repo / "logs" / args.run_name).resolve()
    dst_run_dir.mkdir(parents=True, exist_ok=True)

    dst_g0 = dst_run_dir / "G_0.pth"
    dst_d0 = dst_run_dir / "D_0.pth"

    if dst_g0.exists() and (dst_d0.exists() or not args.pretrained_d) and not args.force_bootstrap:
        print(f"[LoRA] Existing bootstrap checkpoints found under {dst_run_dir}; skip copy")
        return

    if args.pretrained_g:
        src_g0 = _resolve_path(root, args.pretrained_g)
    else:
        src_g0 = (vits_repo / "logs" / args.pretrained_run_name / "G_0.pth").resolve()

    src_d0 = None
    if args.pretrained_d:
        src_d0 = _resolve_path(root, args.pretrained_d)
    elif args.pretrained_run_name:
        candidate = (vits_repo / "logs" / args.pretrained_run_name / "D_0.pth").resolve()
        if candidate.exists():
          src_d0 = candidate

    if not src_g0.exists():
        raise FileNotFoundError(
            "Cannot bootstrap pretrained generator checkpoint. "
            f"Expected G_0 at {src_g0}. "
            "Provide --pretrained-g or correct --pretrained-run-name."
        )

    if args.pretrained_d and (src_d0 is None or not src_d0.exists()):
        raise FileNotFoundError(
            "Cannot bootstrap pretrained discriminator checkpoint. "
            f"Expected D_0 at {src_d0}. "
            "Provide --pretrained-d or correct --pretrained-run-name."
        )

    if args.force_bootstrap or not dst_g0.exists():
        shutil.copy2(src_g0, dst_g0)
        print(f"[LoRA] Bootstrapped G_0: {src_g0} -> {dst_g0}")
    else:
        print(f"[LoRA] Keep existing G_0: {dst_g0}")

    if src_d0 is not None:
        if args.force_bootstrap or not dst_d0.exists():
            shutil.copy2(src_d0, dst_d0)
            print(f"[LoRA] Bootstrapped D_0: {src_d0} -> {dst_d0}")
        else:
            print(f"[LoRA] Keep existing D_0: {dst_d0}")
    else:
        print("[LoRA] No pretrained D_0 provided; D will start from random initialization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA-style VITS training launcher for EmoNarrify")

    # Shared phase3 args
    parser.add_argument("--manifest", type=str, default="data/phase3_esd_manifest.jsonl")
    parser.add_argument("--include-test", type=_str2bool, default=True)
    parser.add_argument("--speaker-filter", type=str, default="0011-0020")
    parser.add_argument(
        "--collapse-speakers-to-one",
        type=_str2bool,
        default=False,
        help="Pass through to phase3_train_vits.py; map all filtered speakers to one sid for pooled single-voice training",
    )
    parser.add_argument("--emotion-labels", type=str, default="angry,happy,neutral,sad,surprise")
    parser.add_argument(
        "--emotion-only",
        type=_str2bool,
        default=False,
        help="Pass through to phase3_train_vits.py; when true sets n_speakers=0 while keeping emotion conditioning on",
    )
    parser.add_argument("--vits-repo", type=str, required=True)
    parser.add_argument("--vits-config", type=str, default="configs/vctk_base.json")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--run-name", type=str, default="emonarrify_phase3_vits_lora")
    parser.add_argument("--work-dir", type=str, default="outputs/vits_phase3_lora")
    parser.add_argument(
        "--stage",
        type=str,
        default="none",
        choices=["none", "stage1", "stage2"],
        help="Two-stage preset: stage1=quality-preserving LoRA warmup, stage2=enable emotion embedding fine-tune",
    )

    # Train overrides
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", type=_str2bool, default=True)
    parser.add_argument("--fp16-run", type=_str2bool, default=True)
    parser.add_argument("--eval-interval", type=int, default=2500)
    parser.add_argument(
        "--text-cleaners",
        type=str,
        default="",
        help="Pass through to phase3_train_vits.py (e.g. english_cleaners2)",
    )
    parser.add_argument(
        "--cleaned-text",
        type=_str2bool,
        default=None,
        help="Pass through to phase3_train_vits.py",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=0,
        help="Pass through to phase3_train_vits.py (0=auto)",
    )

    # LoRA/PEFT settings
    parser.add_argument("--lora-emotion-rank", type=int, default=8)
    parser.add_argument("--lora-emotion-alpha", type=float, default=16.0)
    parser.add_argument("--lora-emotion-dropout", type=float, default=0.05)
    parser.add_argument("--train-only-lora", type=_str2bool, default=True)
    parser.add_argument("--train-emotion-embedding", type=_str2bool, default=True)
    parser.add_argument("--train-speaker-embedding", type=_str2bool, default=False)
    parser.add_argument(
        "--train-conditioning-projections",
        type=_str2bool,
        default=True,
        help="When train_only_lora=true, also train g-conditioning projection layers (cond/cond_layer) to avoid zero-gradient deadlock",
    )

    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument(
        "--bootstrap-pretrained",
        type=_str2bool,
        default=True,
        help="Auto-copy pretrained G_0/D_0 into logs/<run-name> before launching train",
    )
    parser.add_argument(
        "--pretrained-run-name",
        type=str,
        default="emonarrify_phase3_vits_lora",
        help="Source run under vits/logs containing G_0.pth and D_0.pth (should be adapted to 10 speakers for ESD dataset)",
    )
    parser.add_argument("--pretrained-g", type=str, default="", help="Optional explicit source path for G_0.pth")
    parser.add_argument("--pretrained-d", type=str, default="", help="Optional explicit source path for D_0.pth")
    parser.add_argument(
        "--force-bootstrap",
        type=_str2bool,
        default=False,
        help="Overwrite existing logs/<run-name>/G_0.pth and D_0.pth",
    )
    parser.add_argument("--no-launch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    vits_repo = Path(args.vits_repo).expanduser().resolve()

    def _flag_provided(flag: str) -> bool:
        argv = sys.argv[1:]
        return any(tok == flag or tok.startswith(flag + "=") for tok in argv)

    if args.stage == "stage1":
        if not _flag_provided("--train-only-lora"):
            args.train_only_lora = True
        if not _flag_provided("--train-emotion-embedding"):
            args.train_emotion_embedding = False
        if not _flag_provided("--train-speaker-embedding"):
            args.train_speaker_embedding = False
        if not _flag_provided("--train-conditioning-projections"):
            args.train_conditioning_projections = False
        if not _flag_provided("--lora-emotion-rank"):
            args.lora_emotion_rank = 4
        if not _flag_provided("--lora-emotion-alpha"):
            args.lora_emotion_alpha = 8.0
        if not _flag_provided("--learning-rate"):
            args.learning_rate = 2e-5
        if not _flag_provided("--batch-size"):
            args.batch_size = 16
        if not args.text_cleaners.strip():
            args.text_cleaners = "english_cleaners2"
        if args.cleaned_text is None:
            args.cleaned_text = True
        if args.sampling_rate <= 0:
            args.sampling_rate = 22050
        print("[LoRA][Stage1] Applying quality-preserving preset")

    elif args.stage == "stage2":
        if not _flag_provided("--train-only-lora"):
            args.train_only_lora = True
        if not _flag_provided("--train-emotion-embedding"):
            args.train_emotion_embedding = True
        if not _flag_provided("--train-speaker-embedding"):
            args.train_speaker_embedding = False
        if not _flag_provided("--train-conditioning-projections"):
            args.train_conditioning_projections = True
        if not _flag_provided("--lora-emotion-rank"):
            args.lora_emotion_rank = 8
        if not _flag_provided("--lora-emotion-alpha"):
            args.lora_emotion_alpha = 16.0
        if not _flag_provided("--learning-rate"):
            args.learning_rate = 1e-5
        if not _flag_provided("--batch-size"):
            args.batch_size = 16
        if not args.text_cleaners.strip():
            args.text_cleaners = "english_cleaners2"
        if args.cleaned_text is None:
            args.cleaned_text = True
        if args.sampling_rate <= 0:
            args.sampling_rate = 22050
        print("[LoRA][Stage2] Applying emotion-strengthening preset")

    generated_cfg = (root / args.work_dir / "generated_vits_config.json").resolve()

    # 1) Prepare manifests/filelists/config via existing launcher (no training yet)
    prep_cmd = [
        args.python_bin,
        str((root / "phase3_train_vits.py").resolve()),
        "--manifest", args.manifest,
        "--include-test", str(args.include_test).lower(),
        "--speaker-filter", args.speaker_filter,
        "--collapse-speakers-to-one", str(args.collapse_speakers_to_one).lower(),
        "--emotion-conditioning", "true",
        "--emotion-only", str(args.emotion_only).lower(),
        "--emotion-labels", args.emotion_labels,
        "--vits-repo", args.vits_repo,
        "--vits-config", args.vits_config,
        "--python-bin", args.python_bin,
        "--run-name", args.run_name,
        "--work-dir", args.work_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--num-workers", str(args.num_workers),
        "--pin-memory", str(args.pin_memory).lower(),
        "--fp16-run", str(args.fp16_run).lower(),
        "--text-cleaners", args.text_cleaners,
        "--sampling-rate", str(args.sampling_rate),
        "--no-launch",
    ]

    if args.cleaned_text is not None:
        prep_cmd.extend(["--cleaned-text", str(args.cleaned_text).lower()])

    print("[LoRA] Preparing data/config with phase3_train_vits.py")
    print("  " + " ".join(shlex.quote(x) for x in prep_cmd))
    subprocess.run(prep_cmd, check=True)

    if not generated_cfg.exists():
        raise FileNotFoundError(f"Generated config not found: {generated_cfg}")

    # 2) Inject LoRA & PEFT fields
    with generated_cfg.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("train", {})
    cfg.setdefault("model", {})

    cfg["train"]["eval_interval"] = int(args.eval_interval)
    cfg["train"]["train_only_lora"] = bool(args.train_only_lora)
    cfg["train"]["train_emotion_embedding"] = bool(args.train_emotion_embedding)
    cfg["train"]["train_speaker_embedding"] = bool(args.train_speaker_embedding)
    cfg["train"]["train_conditioning_projections"] = bool(args.train_conditioning_projections)

    cfg["model"]["lora_emotion_rank"] = int(args.lora_emotion_rank)
    cfg["model"]["lora_emotion_alpha"] = float(args.lora_emotion_alpha)
    cfg["model"]["lora_emotion_dropout"] = float(args.lora_emotion_dropout)

    with generated_cfg.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print("[LoRA] Patched generated config:", generated_cfg)
    print("[LoRA] train.eval_interval=", cfg["train"]["eval_interval"])
    print("[LoRA] train.train_only_lora=", cfg["train"]["train_only_lora"])
    print("[LoRA] train.train_conditioning_projections=", cfg["train"]["train_conditioning_projections"])
    print(
        "[LoRA] model.lora_emotion_rank/alpha/dropout=",
        cfg["model"]["lora_emotion_rank"],
        cfg["model"]["lora_emotion_alpha"],
        cfg["model"]["lora_emotion_dropout"],
    )

    # 2.5) Ensure run is bootstrapped from pretrained G_0/D_0
    _bootstrap_pretrained_if_needed(args, root=root, vits_repo=vits_repo)

    if args.no_launch:
        print("[LoRA] --no-launch set; exit after config prep.")
        return

    # 3) Launch VITS train_ms.py with patched config
    train_script = (vits_repo / "train_ms.py").resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"train_ms.py not found: {train_script}")

    env = os.environ.copy()
    env["EMONARRIFY_VITS_NUM_WORKERS"] = str(args.num_workers)
    env["EMONARRIFY_VITS_PIN_MEMORY"] = "1" if args.pin_memory else "0"
    env["MASTER_ADDR"] = env.get("MASTER_ADDR", "localhost")
    env["MASTER_PORT"] = str(args.master_port if args.master_port > 0 else find_free_port())

    launch_cmd = [
        args.python_bin,
        str(train_script),
        "-c", str(generated_cfg),
        "-m", args.run_name,
    ]

    print("[LoRA] Launch command:")
    print("  " + " ".join(shlex.quote(x) for x in launch_cmd))
    print(f"[LoRA] Using DDP endpoint: {env['MASTER_ADDR']}:{env['MASTER_PORT']}")

    proc = subprocess.Popen(
        launch_cmd,
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
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"LoRA VITS training exited with code {ret}")


if __name__ == "__main__":
    main()
