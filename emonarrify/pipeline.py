"""
EmoNarrify Pipeline - Hyper Class
Orchestrates Phase 1, 2, 3 into a single end-to-end inference call.
"""

import os
import numpy as np
import torch
import soundfile as sf
from PIL import Image
from typing import Optional

from .config import (
    D_EMO, EMOTION_TO_IDX,
    LOOKUP_TABLE_PATH, SAMPLE_RATE,
)
from .phase1.model import Phase1Model
from .phase2.model import Phase2Model
from .phase3.model import Phase3Model


class EmoNarrifyPipeline:
    """
    End-to-end pipeline: Image -> Emotional Story Audio

    Usage:
        pipeline = EmoNarrifyPipeline()                     # stub mode
        pipeline = EmoNarrifyPipeline(                      # real mode
            phase1_adapter="weights/vlm_lora/",
            phase2_mlp="weights/mlp.pt",
            phase3_tts="weights/tts.pt",
        )
        result = pipeline.run(image)
        pipeline.save_audio(result["audio"], "output.wav")
    """

    def __init__(
        self,
        phase1_adapter: Optional[str] = None,
        phase2_mlp: Optional[str] = None,
        phase3_tts: Optional[str] = None,
        use_fallback: bool = False,
    ):
        """
        Args:
            phase1_adapter: path to VLM LoRA adapter (None = stub)
            phase2_mlp:     path to MLP projection head weights (None = stub)
            phase3_tts:     path to TTS model weights (None = stub)
            use_fallback:   if True, bypass Phase 2 and use discrete label -> lookup table
        """
        self.use_fallback = use_fallback

        print("[EmoNarrify] Initializing pipeline...")
        self.phase1 = Phase1Model(adapter_path=phase1_adapter)
        self.phase3 = Phase3Model(tts_weights_path=phase3_tts)

        if not use_fallback:
            self.phase2 = Phase2Model(mlp_weights_path=phase2_mlp)
        else:
            self.phase2 = None

        # Load lookup table for fallback (always available)
        self._lookup_table = None
        if os.path.exists(LOOKUP_TABLE_PATH):
            self._lookup_table = Phase2Model.load_lookup_table(LOOKUP_TABLE_PATH)

        mode = "STUB" if (phase1_adapter is None and phase3_tts is None) else "REAL"
        fallback_str = " (FALLBACK mode)" if use_fallback else ""
        print(f"[EmoNarrify] Pipeline ready - {mode}{fallback_str}")

    def _get_fallback_embedding(self, emotion_label: str) -> torch.Tensor:
        """Look up emotion embedding from Phase 3's exported table."""
        if self._lookup_table is not None and emotion_label in self._lookup_table:
            return self._lookup_table[emotion_label]

        # If no lookup table available (stub mode), return deterministic one-hot
        emb = torch.zeros(D_EMO)
        idx = EMOTION_TO_IDX.get(emotion_label, 0)
        emb[idx] = 1.0
        return emb

    def run(self, image: Image.Image) -> dict:
        """
        Run full inference pipeline on a single image.

        Args:
            image: PIL.Image input

        Returns:
            dict with keys:
                - "narrative_text": str
                - "emotion_label": str
                - "parse_confidence": str ("full", "partial", or "fallback")
                - "emotion_embedding": torch.Tensor (D_emo,)
                - "audio": np.ndarray (n_samples,)
                - "sample_rate": int
                - "embedding_source": str ("phase2" or "fallback")
        """
        # Step 1: Phase 1 - generate story + emotion label
        phase1_out = self.phase1.generate_story(image)
        narrative_text = phase1_out["narrative_text"]
        emotion_label = phase1_out["emotion_label"]
        parse_confidence = phase1_out.get("parse_confidence", "unknown")

        # Step 2: get emotion embedding
        if not self.use_fallback and self.phase2 is not None:
            try:
                emotion_embedding = self.phase2.encode_image_emotion(image)
                embedding_source = "phase2"
            except Exception as e:
                print(f"[EmoNarrify][Warn] Phase2 failed ({e}); switching to fallback path.")
                emotion_embedding = self._get_fallback_embedding(emotion_label)
                embedding_source = "fallback"
        else:
            emotion_embedding = self._get_fallback_embedding(emotion_label)
            embedding_source = "fallback"

        # Step 3: Phase 3 - synthesize speech
        if embedding_source == "fallback":
            # In fallback mode, Phase 3 follows the MVP contract directly:
            # text + discrete emotion label -> lookup table -> waveform.
            audio = self.phase3.synthesize_from_label(narrative_text, emotion_label)
        else:
            audio = self.phase3.synthesize_speech(narrative_text, emotion_embedding)

        return {
            "narrative_text": narrative_text,
            "emotion_label": emotion_label,
            "parse_confidence": parse_confidence,
            "emotion_embedding": emotion_embedding,
            "audio": audio,
            "sample_rate": SAMPLE_RATE,
            "embedding_source": embedding_source,
        }

    @staticmethod
    def save_audio(audio: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE):
        """Save audio waveform to .wav file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        sf.write(path, audio, sample_rate)
        print(f"[EmoNarrify] Audio saved to {path}")

    def run_and_save(self, image: Image.Image, output_path: str) -> dict:
        """Run pipeline and save audio in one call."""
        result = self.run(image)
        self.save_audio(result["audio"], output_path, result["sample_rate"])
        return result
