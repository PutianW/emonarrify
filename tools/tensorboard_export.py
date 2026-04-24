#!/usr/bin/env python3
"""Launch TensorBoard and export scalar logs to CSV.

Examples
--------
Export scalars from a run directory and launch TensorBoard:
    /home/ubuntu/emonarrify/.venv/bin/python tools/tensorboard_export.py \
      --logdir /home/ubuntu/emonarrify/vits/logs/emonarrify_phase3_vits_from_pretrained_emotion_only_clean \
      --csv /home/ubuntu/emonarrify/outputs/tb_scalars_clean.csv \
      --launch

Only export CSV (no TensorBoard server):
    /home/ubuntu/emonarrify/.venv/bin/python tools/tensorboard_export.py \
      --logdir /home/ubuntu/emonarrify/vits/logs/emonarrify_phase3_vits_from_pretrained_emotion_only_clean \
      --csv /home/ubuntu/emonarrify/outputs/tb_scalars_clean.csv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Iterable, List, Dict, Any


def _import_event_accumulator():
    try:
        from tensorboard.backend.event_processing import event_accumulator  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "tensorboard package is required. Install it with: pip install tensorboard"
        ) from exc
    return event_accumulator


def discover_event_files(logdir: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = sorted(logdir.rglob("events.out.tfevents.*"))
    else:
        files = sorted(logdir.glob("events.out.tfevents.*"))
    return [p for p in files if p.is_file()]


def extract_scalars(event_files: Iterable[Path]) -> List[Dict[str, Any]]:
    event_accumulator = _import_event_accumulator()

    rows: List[Dict[str, Any]] = []
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.AUDIO: 0,
                event_accumulator.HISTOGRAMS: 0,
            },
        )
        ea.Reload()

        tags = ea.Tags().get("scalars", [])
        for tag in tags:
            for point in ea.Scalars(tag):
                rows.append(
                    {
                        "event_file": str(event_file),
                        "tag": tag,
                        "step": int(point.step),
                        "wall_time": float(point.wall_time),
                        "value": float(point.value),
                    }
                )

    rows.sort(key=lambda r: (r["tag"], r["step"], r["wall_time"]))
    return rows


def write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["event_file", "tag", "step", "wall_time", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)


def launch_tensorboard(logdir: Path, host: str, port: int, open_browser: bool) -> int:
    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--host",
        host,
        "--port",
        str(port),
    ]

    url = f"http://{host}:{port}"
    print(f"[TensorBoard] starting: {' '.join(cmd)}")
    print(f"[TensorBoard] url: {url}")

    proc = subprocess.Popen(cmd)
    if open_browser:
        time.sleep(1.0)
        webbrowser.open(url)

    print("[TensorBoard] Press Ctrl+C to stop.")
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 130


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalar logs to CSV and optionally launch TensorBoard"
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        required=True,
        help="Directory containing events.out.tfevents.* files",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path (if omitted, CSV export is skipped)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for event files under --logdir",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch TensorBoard after CSV export",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open browser automatically when launching TensorBoard",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logdir = args.logdir.expanduser().resolve()
    if not logdir.exists():
        raise FileNotFoundError(f"logdir not found: {logdir}")

    event_files = discover_event_files(logdir, recursive=args.recursive)
    if not event_files:
        raise RuntimeError(
            f"No event files found in {logdir}. "
            "Expected files like events.out.tfevents.*"
        )

    print(f"[Info] Found {len(event_files)} event files")
    for p in event_files:
        print(f"  - {p}")

    if args.csv is not None:
        rows = extract_scalars(event_files)
        write_csv(rows, args.csv.expanduser().resolve())
        print(f"[CSV] wrote {len(rows)} scalar rows to: {args.csv.expanduser().resolve()}")

    if args.launch:
        exit_code = launch_tensorboard(
            logdir=logdir,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
        )
        if exit_code not in (0, 130):
            raise RuntimeError(f"TensorBoard exited with code {exit_code}")


if __name__ == "__main__":
    main()
