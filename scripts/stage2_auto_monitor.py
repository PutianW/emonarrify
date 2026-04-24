from __future__ import annotations

import argparse
import json
import subprocess
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


EMOTIONS_DEFAULT = ["angry", "happy", "neutral", "sad", "surprise"]


def _iter_of(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return -1


def _load_lora_alpha(model_dir: Path, default: float = 16.0) -> float:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return default
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return float(cfg.get("model", {}).get("lora_emotion_alpha", default))
    except Exception:
        return default


def _tensor_stats(t: torch.Tensor | None) -> Dict:
    if t is None:
        return {"missing": True}
    tf = t.float().flatten()
    return {
        "shape": list(t.shape),
        "mean": float(tf.mean()),
        "std": float(tf.std()),
        "maxabs": float(tf.abs().max()),
        "l2": float(torch.linalg.norm(tf, ord=2)),
        "nonzero": int((tf != 0).sum()),
        "numel": int(tf.numel()),
    }


def _pairwise_dist_stats(m: np.ndarray) -> Dict[str, float]:
    if m.shape[0] < 2:
        return {"mean_pairwise_l2": 0.0, "min_pairwise_l2": 0.0, "max_pairwise_l2": 0.0}
    dists = [float(np.linalg.norm(m[i] - m[j])) for i, j in combinations(range(m.shape[0]), 2)]
    return {
        "mean_pairwise_l2": float(np.mean(dists)),
        "min_pairwise_l2": float(np.min(dists)),
        "max_pairwise_l2": float(np.max(dists)),
    }


def _pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    comp = x @ vt[:2].T
    var = (s ** 2) / max(x.shape[0] - 1, 1)
    evr = var / (var.sum() + 1e-12)
    return comp, evr


def _render_pca(
    ckpt_name: str,
    iteration: int | None,
    emb_e: np.ndarray | None,
    eff: np.ndarray | None,
    emotions: List[str],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if emb_e is not None:
        pts, evr = _pca2(emb_e)
        axes[0].scatter(pts[:, 0], pts[:, 1], c=np.arange(len(pts)), cmap="tab10", s=70)
        for i, n in enumerate(emotions[: len(pts)]):
            axes[0].annotate(n, (pts[i, 0], pts[i, 1]))
        axes[0].set_title(f"emb_e PCA2 (EVR1={evr[0]:.3f}, EVR2={evr[1] if len(evr)>1 else 0:.3f})")
    else:
        axes[0].text(0.5, 0.5, "emb_e missing", ha="center", va="center")
        axes[0].set_title("emb_e PCA2")

    if eff is not None:
        pts2, evr2 = _pca2(eff)
        axes[1].scatter(pts2[:, 0], pts2[:, 1], c=np.arange(len(pts2)), cmap="tab10", s=70)
        for i, n in enumerate(emotions[: len(pts2)]):
            axes[1].annotate(n, (pts2[i, 0], pts2[i, 1]))
        axes[1].set_title(f"effective PCA2 (EVR1={evr2[0]:.3f}, EVR2={evr2[1] if len(evr2)>1 else 0:.3f})")
    else:
        axes[1].text(0.5, 0.5, "effective missing", ha="center", va="center")
        axes[1].set_title("effective PCA2")

    for ax in axes:
        ax.axhline(0, linewidth=0.5)
        ax.axvline(0, linewidth=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{ckpt_name} | iter={iteration}")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _run_inference_for_ckpt(
    python_bin: str,
    project_root: Path,
    run_name: str,
    ckpt: Path,
    metadata: Path,
    out_dir: Path,
    text: str,
    emotions: List[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for emo in emotions:
        out_wav = out_dir / f"{emo}.wav"
        cmd = [
            python_bin,
            str(project_root / "phase3_infer_vits.py"),
            "--text",
            text,
            "--emotion",
            emo,
            "--run-name",
            run_name,
            "--ckpt",
            str(ckpt),
            "--metadata",
            str(metadata),
            "--output",
            str(out_wav),
        ]
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto monitor Stage2 checkpoints: inference + PCA + separation stats")
    parser.add_argument("--project-root", type=str, default="/home/ubuntu/emonarrify")
    parser.add_argument("--run-name", type=str, default="emonarrify_phase3_vits_ljs_emotion_s2")
    parser.add_argument("--metadata", type=str, default="outputs/vits_phase3_ljs_emotion_s2/launcher_metadata.json")
    parser.add_argument("--text", type=str, default="We are going to have apresentation tomorrow")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--start-step", type=int, default=1000)
    parser.add_argument("--step-interval", type=int, default=1000)
    parser.add_argument("--python-bin", type=str, default="/home/ubuntu/emonarrify/.venv/bin/python")
    parser.add_argument("--out-root", type=str, default="outputs/vits_ckpt_compare_auto/s2_auto")
    parser.add_argument("--state-json", type=str, default="outputs/vits_phase3_ljs_emotion_s2/auto_monitor_state.json")
    parser.add_argument("--summary-json", type=str, default="outputs/vits_phase3_ljs_emotion_s2/embedding_inspect/embedding_summary_auto.json")
    parser.add_argument("--emotions", type=str, default=",".join(EMOTIONS_DEFAULT))
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    model_dir = project_root / "vits" / "logs" / args.run_name
    metadata = (project_root / args.metadata).resolve() if not Path(args.metadata).is_absolute() else Path(args.metadata).resolve()
    out_root = (project_root / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root).resolve()
    state_json = (project_root / args.state_json).resolve() if not Path(args.state_json).is_absolute() else Path(args.state_json).resolve()
    summary_json = (project_root / args.summary_json).resolve() if not Path(args.summary_json).is_absolute() else Path(args.summary_json).resolve()
    emotions = [x.strip() for x in args.emotions.split(",") if x.strip()]

    state_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    if state_json.exists():
        try:
            with state_json.open("r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {"processed": []}
    else:
        state = {"processed": []}

    processed = set(state.get("processed", []))

    if summary_json.exists():
        try:
            with summary_json.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = []
    else:
        summary = []

    lora_alpha = _load_lora_alpha(model_dir, default=16.0)
    print(f"[monitor] start | model_dir={model_dir} | lora_alpha={lora_alpha}", flush=True)

    while True:
        ckpts = sorted(model_dir.glob("G_*.pth"), key=_iter_of)
        targets = []
        for ck in ckpts:
            step = _iter_of(ck)
            if step < 0:
                continue
            if step < args.start_step:
                continue
            if step % args.step_interval != 0:
                continue
            if ck.name in processed:
                continue
            targets.append(ck)

        for ck in targets:
            step = _iter_of(ck)
            print(f"[monitor] processing {ck.name}", flush=True)
            try:
                obj = torch.load(ck, map_location="cpu")
                st = obj["model"]
                iteration = obj.get("iteration")

                emb = st.get("emb_e.weight")
                A = st.get("emb_e_lora_A.weight")
                B = st.get("emb_e_lora_B.weight")

                eff = None
                if emb is not None and A is not None and B is not None:
                    rank = max(int(A.shape[1]), 1)
                    scale = float(lora_alpha) / float(rank)
                    eff = emb.float() + (A.float() @ B.float().T) * scale

                emb_np = emb.detach().cpu().numpy() if emb is not None else None
                eff_np = eff.detach().cpu().numpy() if eff is not None else None

                pca_png = summary_json.parent / f"{ck.stem}_pca.png"
                _render_pca(ck.name, iteration, emb_np, eff_np, emotions, pca_png)

                # Run inference
                infer_dir = out_root / ck.stem
                _run_inference_for_ckpt(
                    python_bin=args.python_bin,
                    project_root=project_root,
                    run_name=args.run_name,
                    ckpt=ck,
                    metadata=metadata,
                    out_dir=infer_dir,
                    text=args.text,
                    emotions=emotions,
                )

                row = {
                    "checkpoint": str(ck),
                    "step": step,
                    "iteration": int(iteration) if iteration is not None else None,
                    "emb_e": _tensor_stats(emb),
                    "emb_e_lora_A": _tensor_stats(A),
                    "emb_e_lora_B": _tensor_stats(B),
                    "effective_embedding": _tensor_stats(eff),
                    "emb_pairwise_l2": _pairwise_dist_stats(emb_np) if emb_np is not None else None,
                    "effective_pairwise_l2": _pairwise_dist_stats(eff_np) if eff_np is not None else None,
                    "pca_png": str(pca_png),
                    "inference_dir": str(infer_dir),
                }

                summary.append(row)
                with summary_json.open("w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)

                processed.add(ck.name)
                state["processed"] = sorted(processed, key=lambda x: _iter_of(Path(x)))
                with state_json.open("w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)

                print(
                    "[monitor] done {} | emb_mean_l2={:.6f} | eff_mean_l2={:.6f}".format(
                        ck.name,
                        (row["emb_pairwise_l2"] or {}).get("mean_pairwise_l2", 0.0),
                        (row["effective_pairwise_l2"] or {}).get("mean_pairwise_l2", 0.0),
                    ),
                    flush=True,
                )
            except Exception as exc:
                print(f"[monitor] ERROR on {ck.name}: {exc}", flush=True)

        time.sleep(max(args.poll_seconds, 5))


if __name__ == "__main__":
    main()
