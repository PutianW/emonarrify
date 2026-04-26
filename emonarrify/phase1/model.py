"""
Phase 1: Vision-to-Text with Emotion (VLM Instruction Tuning)
Owner: Cassie Chang

Interface contract:
    generate_story(image: PIL.Image) -> dict
    Returns {"narrative_text": str, "emotion_label": str, "parse_confidence": str}
    parse_confidence: "full" (both tags parsed), "partial" (one tag failed), "fallback" (both failed)
"""

import torch
import json
import re
from PIL import Image
import os

# Heavy deps (unsloth, transformers, datasets) are deferred to their use
# sites so this module can be imported in stub mode without those packages
# installed (e.g. e2e pipeline smoke without the LoRA adapter).
# `datasets.load_dataset` was imported at top level but never used here
# in this file -- removed during lazy-import hygiene pass.

# from ..config import EMOTION_LABELS, DEFAULT_EMOTION

# logger = logging.getLogger(__name__)


INSTRUCTION = (
    "You must use the image to answer.\n"
    # "If the answer cannot be seen in the image, do not say it.\n\n"
    "First describe the image briefly.\n"
    # "Then predict emotion.\n\n"
    "Then choose one emotion from: neutral, happy, angry, sad, surprise.\n\n"
)


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
        self._tokenizer = None
        self._is_stub = adapter_path is None

        if not self._is_stub:
            self._load_model()

    def _load_model(self):
        """Load fine-tuned VLM with LoRA adapter.

        unsloth import is deferred here so module import does not require
        the unsloth dep when only stub mode is used (e.g. e2e pipeline
        smoke without the LoRA adapter).
        """
        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name = self.adapter_path, # YOUR MODEL YOU USED FOR TRAINING
            load_in_4bit = True, # Set to False for 16bit LoRA
        )
        FastVisionModel.for_inference(model) # Enable for inference!
        
        self._model = model
        self._tokenizer = tokenizer

        return model
    
    def _load_emotion_clf():
        from transformers import pipeline
        # print("Loading emotion classifier...")
        clf = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
            device=-1
        )
        return clf

    def _map_emotion(labels):
        label_scores = {l["label"]: l["score"] for l in labels}
        top_label = max(label_scores, key=label_scores.get)

        if top_label in ["joy", "amusement", "love", "excitement", "approval", "optimism", "admiration"]:
            return "happy"
        elif top_label in ["anger", "annoyance", "disapproval"]:
            return "angry"
        elif top_label in ["sadness", "grief", "disappointment", "fear", "remorse"]:
            return "sad"
        elif top_label in ["surprise", "realization"]:
            return "surprise"
        else:
            return "neutral"

    def _parse_output(self, raw_text: str) -> dict:
        """Parse structured VLM output into narrative_text and emotion_label."""
        raw_text = raw_text.replace("<|im_end|>", "").strip()

        desc_match = re.search(r"Description:\s*(.*)", raw_text)
        emo_match  = re.search(r"Emotion:\s*(.*)", raw_text)

        description = desc_match.group(1).strip() if desc_match else ""
        emotion = emo_match.group(1).strip() if emo_match else ""

        return description, emotion

    def generate_story(self, image_path: Image.Image, emotion_label=None) -> dict:
        """
        Generate narrative text and emotion label from an image.

        Args:
            image: PIL.Image input

        Returns:
            {"narrative_text": str, "emotion_label": str, "parse_confidence": str}
        """
        if self._is_stub:
            return {
                "narrative_text": "A quiet scene unfolds, captured in this image, telling its silent story.",
                "emotion_label": "neutral",
                "parse_confidence": "fallback",
            }

        image = Image.open(image_path)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION}
            ]}
        ]

        input_text = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._tokenizer(
            image,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")

        from transformers import TextStreamer
        text_streamer = TextStreamer(self._tokenizer, skip_prompt = True)
        output_ids = self._model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                    use_cache = True, temperature = 1.5, min_p = 0.1)

        output_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=False)
        description, emotion = self._parse_output(output_text)

        if emotion_label is None:
            clf = _load_emotion_clf()
            outputs = clf(description)[0]
            emotion_label = _map_emotion(outputs)

        return {
            "image_path": image_path,
            "narrative_text": description,
            "emotion_label": emotion_label,
        }


# =========================================================================
# Convenience function (matches interface contract)
# =========================================================================
_default_model: Phase1Model = None


def generate_story(image_path: str, emotion_label: str) -> dict:
    """Top-level interface function as specified in the project contract."""
    global _default_model
    if _default_model is None:
        _default_model = Phase1Model(adapter_path="weights/qwen_lora")  # stub by default
    return _default_model.generate_story(image_path, emotion_label)
