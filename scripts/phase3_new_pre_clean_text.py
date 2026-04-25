"""Pre-clean Phase 3 (new) raw filelists for cleaned_text training mode.

For each line `audio|eid|raw_text` in data/filelists/phase3_new_{train,val,test}.txt,
runs the configured cleaner pipeline (default english_cleaners2 →
phonemizer → espeak-ng) once and writes `audio|eid|phonemes` to the
`.cleaned.txt` sibling. This bypasses phonemizer at training time
(loader takes the cleaned_text=True branch), eliminating the multi-worker
phonemizer subprocess deadlock observed when launching with num_workers=4
in tmux detached mode.

Single-process serial — phonemizer's espeak-ng subprocess backend does not
parallelize cleanly across workers. ESD English subset (17,500 utts) takes
~12-15 min on a single thread.

Idempotent: skips a split if its `.cleaned.txt` already exists and is
non-empty unless `--force` is passed.

Output schema mirrors upstream LJS/VCTK cleaned filelist convention
(IPA phoneme string, not symbol IDs):
  audio_path|eid|ɪt hɐz jˈuːzd ˈʌðɚ tɹˈɛʒɚɹi lˈɔː ɛnfˈoːɹsmənt ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("phonemizer").setLevel(logging.ERROR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vits"))

from phonemizer.backend import EspeakBackend  # noqa: E402
from text.cleaners import (  # noqa: E402
    convert_to_ascii, lowercase, expand_abbreviations, collapse_whitespace,
)


# Persistent espeak-ng backend instance reused across all utterances.
# vits/text/cleaners.py english_cleaners2() calls phonemize() (function-level
# API) which spawns a new EspeakBackend per call; the resulting espeak-ng
# subprocess churn leaks heavily — observed 26GB RSS at ~7000 utts before
# throughput collapsed from 44 utts/s to 1.5 utts/s. Single backend instance
# avoids the leak and keeps steady ~40 utts/s throughout.
_espeak_backend: EspeakBackend | None = None


def _get_backend() -> EspeakBackend:
    global _espeak_backend
    if _espeak_backend is None:
        _espeak_backend = EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True,
        )
    return _espeak_backend


def english_cleaners2_persistent(text: str) -> str:
    """Mirror of vits/text/cleaners.py english_cleaners2 but with shared backend."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes_list = _get_backend().phonemize([text], strip=True)
    return collapse_whitespace(phonemes_list[0])


_CLEANER_REGISTRY = {
    "english_cleaners2": english_cleaners2_persistent,
}


def _clean_text_persistent(text: str, cleaner_names: list[str]) -> str:
    for name in cleaner_names:
        if name in _CLEANER_REGISTRY:
            text = _CLEANER_REGISTRY[name](text)
        else:
            # Fallback to the upstream cleaner (no leak protection — only
            # english_cleaners2 known to invoke phonemize)
            from text.cleaners import __dict__ as _cleaners_ns
            text = _cleaners_ns[name](text)
    return text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="vits/configs/phase3_new_v1.json",
                   help="Read text_cleaners list from this hps.data block")
    p.add_argument("--filelist-dir", default="data/filelists")
    p.add_argument("--force", action="store_true",
                   help="Re-clean even if .cleaned.txt exists and is non-empty")
    return p.parse_args()


def clean_one_filelist(src_path: str, dst_path: str, cleaners: list[str], force: bool) -> dict:
    if not force and os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        n_lines = sum(1 for _ in open(dst_path))
        return {"skipped": True, "n_lines": n_lines}

    n_in = n_out = n_failed = 0
    t0 = time.time()
    with open(src_path) as fin, open(dst_path, "w") as fout:
        for raw_line in fin:
            raw_line = raw_line.rstrip("\n")
            if not raw_line:
                continue
            n_in += 1
            try:
                audio, eid, raw_text = raw_line.split("|", 2)
            except ValueError:
                raise ValueError(f"Malformed line in {src_path}: {raw_line!r}")
            try:
                cleaned = _clean_text_persistent(raw_text, cleaners)
            except Exception as exc:
                n_failed += 1
                print(f"  [warn] failed to clean line {n_in}: {exc}; raw_text={raw_text!r}")
                continue
            fout.write(f"{audio}|{eid}|{cleaned}\n")
            n_out += 1
            if n_in % 1000 == 0:
                elapsed = time.time() - t0
                rate = n_in / max(elapsed, 1e-3)
                print(f"  [{os.path.basename(src_path)}] {n_in} done ({rate:.1f} utts/s, "
                      f"elapsed {elapsed:.1f}s)")
    return {"skipped": False, "n_in": n_in, "n_out": n_out, "n_failed": n_failed,
            "wall_seconds": time.time() - t0}


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    cleaners = cfg["data"]["text_cleaners"]
    print(f"[pre-clean] config: {args.config}")
    print(f"[pre-clean] cleaners: {cleaners}")
    print(f"[pre-clean] filelist_dir: {args.filelist_dir}")

    splits = ["train", "val", "test"]
    summaries: dict[str, dict] = {}
    t_total = time.time()
    for split in splits:
        src = os.path.join(args.filelist_dir, f"phase3_new_{split}.txt")
        dst = os.path.join(args.filelist_dir, f"phase3_new_{split}.cleaned.txt")
        print(f"\n[pre-clean] {split}: {src} -> {dst}")
        r = clean_one_filelist(src, dst, cleaners, args.force)
        summaries[split] = r
        if r["skipped"]:
            print(f"  [skipped] already exists with {r['n_lines']} lines (use --force to redo)")
        else:
            rate = r["n_out"] / max(r["wall_seconds"], 1e-3)
            print(f"  [done] {r['n_out']}/{r['n_in']} lines in {r['wall_seconds']:.1f}s "
                  f"({rate:.1f} utts/s), failed={r['n_failed']}")
    print(f"\n[pre-clean] total wall time: {time.time() - t_total:.1f}s")
    print("\n[pre-clean] sanity head -3 of each .cleaned.txt:")
    for split in splits:
        dst = os.path.join(args.filelist_dir, f"phase3_new_{split}.cleaned.txt")
        if not os.path.exists(dst):
            continue
        print(f"  --- {dst} ---")
        with open(dst) as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"    {line.rstrip()}")


if __name__ == "__main__":
    main()
