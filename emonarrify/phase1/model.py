"""
Phase 1: Vision-to-Text with Emotion (VLM Instruction Tuning)
Owner: Member B

Interface contract:
    generate_story(image: PIL.Image) -> dict
    Returns {"narrative_text": str, "emotion_label": str, "parse_confidence": str}
    parse_confidence: "full" (both tags parsed), "partial" (one tag failed), "fallback" (both failed)
"""

import re
import logging
from PIL import Image

from ..config import EMOTION_LABELS, DEFAULT_EMOTION

logger = logging.getLogger(__name__)


class Phase1Model:
    """VLM wrapper for story generation + emotion labeling."""

    def __init__(self, adapter_path: str = None):
        """
        Args:
            adapter_path: path to fine-tuned LoRA adapter weights.
                          If None, runs in stub mode (returns dummy data).
        """
        self.adapter_path = adapter_path
        self._model = None
        self._is_stub = adapter_path is None

        if not self._is_stub:
            self._load_model()

    def _load_model(self):
        """Load fine-tuned VLM with LoRA adapter. Member B implements this."""
        # TODO: Member B - load your VLM backbone + LoRA adapter here
        # Example:
        #   self._model = load_vlm(self.adapter_path)
        raise NotImplementedError("Member B: implement VLM loading")

    def _parse_output(self, raw_text: str) -> dict:
        """Parse structured VLM output into narrative_text and emotion_label."""
        story_match = re.search(r"\[STORY\](.*?)\[/STORY\]", raw_text, re.DOTALL)
        emotion_match = re.search(r"\[EMOTION\](.*?)\[/EMOTION\]", raw_text, re.DOTALL)

        parse_confidence = "full"

        if story_match:
            narrative_text = story_match.group(1).strip()
        else:
            logger.warning("Phase 1: [STORY] tag parse failed, using raw text as fallback")
            narrative_text = raw_text.strip()
            parse_confidence = "partial"

        emotion_label = DEFAULT_EMOTION
        if emotion_match:
            candidate = emotion_match.group(1).strip().lower()
            if candidate in EMOTION_LABELS:
                emotion_label = candidate
            else:
                logger.warning(f"Phase 1: emotion '{candidate}' not in label set, falling back to '{DEFAULT_EMOTION}'")
                parse_confidence = "partial"
        else:
            logger.warning(f"Phase 1: [EMOTION] tag parse failed, falling back to '{DEFAULT_EMOTION}'")
            parse_confidence = "fallback" if parse_confidence == "partial" else "partial"

        return {
            "narrative_text": narrative_text,
            "emotion_label": emotion_label,
            "parse_confidence": parse_confidence,
        }

    def generate_story(self, image: Image.Image) -> dict:
        """
        Generate narrative text and emotion label from an image.

        Args:
            image: PIL.Image input

        Returns:
            {"narrative_text": str, "emotion_label": str}
        """
        if self._is_stub:
            return {
                "narrative_text": "A quiet scene unfolds under a pale sky, carrying a sense of calm solitude.",
                "emotion_label": "neutral",
                "parse_confidence": "full",
            }

        # TODO: Member B - implement VLM inference here
        # raw_output = self._model.generate(image, prompt=INSTRUCTION_PROMPT)
        # return self._parse_output(raw_output)
        raise NotImplementedError("Member B: implement VLM inference")


# =========================================================================
# Convenience function (matches interface contract)
# =========================================================================
_default_model: Phase1Model = None


def generate_story(image: Image.Image) -> dict:
    """Top-level interface function as specified in the project contract."""
    global _default_model
    if _default_model is None:
        _default_model = Phase1Model(adapter_path=None)  # stub by default
    return _default_model.generate_story(image)
