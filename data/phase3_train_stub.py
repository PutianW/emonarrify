"""Run Phase 3 training stub from a prepared manifest."""

import argparse

from emonarrify.phase3.train import train_phase3_stub


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 stub training")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/phase3_esd_manifest.jsonl",
        help="Phase 3 JSONL manifest path",
    )
    parser.add_argument(
        "--lookup-out",
        type=str,
        default="weights/emotion_lookup_table.json",
        help="Output path for exported lookup table JSON",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_phase3_stub(
        manifest_path=args.manifest,
        lookup_out_path=args.lookup_out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()