"""Evaluate (summarize) Phase 3 manifest distribution."""

import argparse
import json

from emonarrify.phase3.evaluation import summarize_phase3_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase3 manifest summary")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/phase3_esd_manifest.jsonl",
        help="Path to phase3 manifest JSONL",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional output path to save summary JSON",
    )
    args = parser.parse_args()

    summary = summarize_phase3_manifest(args.manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to: {args.save_json}")


if __name__ == "__main__":
    main()