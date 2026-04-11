"""
EmoNarrify - CLI Inference Entry Point
Usage:
    python run_inference.py --image path/to/image.jpg --output outputs/result.wav
    python run_inference.py --image path/to/image.jpg --fallback   # use fallback mode
"""

import argparse
import os

from PIL import Image

from emonarrify.pipeline import EmoNarrifyPipeline


def main():
    parser = argparse.ArgumentParser(description="EmoNarrify Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="outputs/output.wav", help="Output .wav path")
    parser.add_argument("--phase1", type=str, default=None, help="Path to VLM LoRA adapter")
    parser.add_argument("--phase2", type=str, default=None, help="Path to MLP weights")
    parser.add_argument("--phase3", type=str, default=None, help="Path to TTS weights")
    parser.add_argument("--fallback", action="store_true", help="Use fallback (skip Phase 2)")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")

    pipeline = EmoNarrifyPipeline(
        phase1_adapter=args.phase1,
        phase2_mlp=args.phase2,
        phase3_tts=args.phase3,
        use_fallback=args.fallback,
    )

    result = pipeline.run_and_save(image, args.output)

    print(f"\n{'='*60}")
    print(f"Story:    {result['narrative_text'][:100]}...")
    print(f"Emotion:  {result['emotion_label']}")
    print(f"Source:   {result['embedding_source']}")
    print(f"Audio:    {args.output} ({len(result['audio'])} samples)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
