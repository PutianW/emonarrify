"""
Phase 3: Emotion-Conditioned TTS Fine-Tuning
Owner: Member A

Interface contract:
    synthesize_speech(text: str, emotion_embedding: torch.Tensor) -> np.ndarray
    Returns audio waveform, shape (n_samples,)

This phase trains first and exports:
    1. Fine-tuned TTS model weights
    2. Emotion lookup table (JSON) for Phase 2
"""

import json
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from ..config import (
    D_EMO, NUM_EMOTIONS, EMOTION_LABELS, EMOTION_TO_IDX,
    SAMPLE_RATE, LOOKUP_TABLE_PATH,
)


class EmotionLookupTable(nn.Module):
    """Learnable emotion embedding table: 5 emotions -> D_emo-dim vectors."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(NUM_EMOTIONS, D_EMO)

    def forward(self, emotion_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emotion_idx: int tensor, index into EMOTION_LABELS
        Returns:
            torch.Tensor of shape (D_emo,) or (batch, D_emo)
        """
        return self.embedding(emotion_idx)

    def export_json(self, path: str = LOOKUP_TABLE_PATH):
        """Export trained embeddings as JSON for Phase 2 consumption."""
        data = {
            "embedding_dim": D_EMO,
            "embeddings": {},
        }
        with torch.no_grad():
            for label in EMOTION_LABELS:
                idx = torch.tensor(EMOTION_TO_IDX[label])
                vec = self.embedding(idx).cpu().tolist()
                data["embeddings"][label] = vec

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Lookup table exported to {path}")

    def cosine_similarity_matrix(self) -> np.ndarray:
        """Compute pairwise cosine similarity across all 5 emotion embeddings."""
        with torch.no_grad():
            all_indices = torch.arange(NUM_EMOTIONS)
            embeddings = self.embedding(all_indices)  # (5, D_emo)
            normed = embeddings / embeddings.norm(dim=1, keepdim=True)
            sim = (normed @ normed.T).cpu().numpy()
        return sim


class Phase3Model:
    """Emotion-conditioned TTS wrapper."""

    def __init__(self, tts_weights_path: str = None):
        """
        Args:
            tts_weights_path: path to fine-tuned TTS model weights.
                              If None, runs in stub mode.
        """
        self.tts_weights_path = tts_weights_path
        self._is_stub = tts_weights_path is None
        self._tts_model = None
        self._lookup_table = EmotionLookupTable()

        if not self._is_stub:
            self._load_model()

    def _load_model(self):
        """Load fine-tuned TTS model. Member A implements this."""
        # TODO: Member A - load your TTS backbone + emotion conditioning here
        # Example:
        #   self._tts_model = load_tts(self.tts_weights_path)
        #   self._lookup_table.load_state_dict(torch.load("emotion_table.pt"))
        raise NotImplementedError("Member A: implement TTS model loading")

    def synthesize_speech(
        self,
        text: str,
        emotion_embedding: torch.Tensor,
    ) -> np.ndarray:
        """
        Synthesize emotionally expressive speech.

        Args:
            text:              narrative text to speak
            emotion_embedding: torch.Tensor of shape (D_emo,), from Phase 2

        Returns:
            np.ndarray audio waveform, shape (n_samples,)
        """
        assert emotion_embedding.shape == (D_EMO,), (
            f"Expected embedding shape ({D_EMO},), got {emotion_embedding.shape}"
        )

        if self._is_stub:
            duration_sec = max(1.0, len(text) * 0.05)
            n_samples = int(duration_sec * SAMPLE_RATE)
            return np.zeros(n_samples, dtype=np.float32)

        # TODO: Member A - implement TTS inference here
        # audio = self._tts_model.synthesize(text, emotion_embedding)
        # return audio
        raise NotImplementedError("Member A: implement TTS synthesis")


# =========================================================================
# Convenience function (matches interface contract)
# =========================================================================
_default_model: Phase3Model = None


def synthesize_speech(text: str, emotion_embedding: torch.Tensor) -> np.ndarray:
    """Top-level interface function as specified in the project contract."""
    global _default_model
    if _default_model is None:
        _default_model = Phase3Model(tts_weights_path=None)  # stub by default
    return _default_model.synthesize_speech(text, emotion_embedding)
