"""Prepare ESD manifest for Phase 3 training.

Example:
    python data/phase3_prepare_data.py --esd-root ESD
"""

import argparse

from emonarrify.phase3.data import build_phase3_esd_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Phase 3 ESD JSONL manifest")
    parser.add_argument("--esd-root", type=str, default="ESD", help="Path to ESD root directory")
    parser.add_argument(
        "--output",
        type=str,
        default="data/phase3_esd_manifest.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation speaker ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test speaker ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    build_phase3_esd_manifest(
        esd_root=args.esd_root,
        output_jsonl=args.output,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()