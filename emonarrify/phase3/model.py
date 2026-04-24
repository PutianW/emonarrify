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
import os
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

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Lookup table exported to {path}")

    def load_json(self, path: str = LOOKUP_TABLE_PATH):
        """Load lookup table vectors from JSON exported by Phase 3 training."""
        with open(path, "r") as f:
            data = json.load(f)

        if data.get("embedding_dim") != D_EMO:
            raise ValueError(
                f"Lookup table dim {data.get('embedding_dim')} != config D_EMO {D_EMO}"
            )

        expected = set(EMOTION_LABELS)
        found = set(data.get("embeddings", {}).keys())
        if expected != found:
            raise ValueError(
                f"Lookup table labels mismatch. expected={sorted(expected)}, found={sorted(found)}"
            )

        weight = torch.zeros(NUM_EMOTIONS, D_EMO, dtype=self.embedding.weight.dtype)
        for label in EMOTION_LABELS:
            vec = data["embeddings"][label]
            if len(vec) != D_EMO:
                raise ValueError(f"Label {label!r} vector dim {len(vec)} != {D_EMO}")
            weight[EMOTION_TO_IDX[label]] = torch.tensor(vec, dtype=weight.dtype)

        with torch.no_grad():
            self.embedding.weight.copy_(weight)

    def get_embedding_by_label(self, emotion_label: str) -> torch.Tensor:
        """Get one 128D embedding from a canonical emotion label."""
        label = emotion_label.strip().lower()
        if label not in EMOTION_TO_IDX:
            raise ValueError(
                f"Unknown emotion label: {emotion_label!r}. Expected one of {EMOTION_LABELS}."
            )
        idx = torch.tensor(EMOTION_TO_IDX[label], dtype=torch.long)
        with torch.no_grad():
            return self.embedding(idx)

    def cosine_similarity_matrix(self) -> np.ndarray:
        """Compute pairwise cosine similarity across all 5 emotion embeddings."""
        with torch.no_grad():
            all_indices = torch.arange(NUM_EMOTIONS)
            embeddings = self.embedding(all_indices)  # (5, D_emo)
            normed = embeddings / embeddings.norm(dim=1, keepdim=True)
            sim = (normed @ normed.T).cpu().numpy()
        return sim


class ConditionalNeuralTTSBackbone(nn.Module):
    """Lightweight GRU-based emotion-conditioned neural TTS backbone (MVP)."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, vocab_size: int = 512):
        super().__init__()
        self.sample_rate = sample_rate
        self.vocab_size = vocab_size
        self.d_model = 192

        self.text_embedding = nn.Embedding(vocab_size, self.d_model)
        self.text_encoder = nn.GRU(
            input_size=self.d_model,
            hidden_size=160,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.text_proj = nn.Sequential(
            nn.Linear(320, self.d_model),
            nn.SiLU(),
            nn.LayerNorm(self.d_model),
        )
        self.emo_proj = nn.Sequential(
            nn.Linear(D_EMO, self.d_model),
            nn.SiLU(),
            nn.LayerNorm(self.d_model),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.SiLU(),
            nn.LayerNorm(self.d_model),
        )

        # [pitch, rate, energy, noise, vibrato, brightness]
        self.prosody_head = nn.Linear(self.d_model, 6)
        # 6 harmonic amplitudes
        self.harmonic_head = nn.Linear(self.d_model, 6)

    @staticmethod
    def _text_to_ids(text: str, vocab_size: int) -> torch.Tensor:
        text = (text or "").strip()
        if not text:
            text = "..."

        ids = []
        for ch in text:
            o = ord(ch)
            if 32 <= o <= 126:
                ids.append(min(o - 31, vocab_size - 1))
            else:
                ids.append(128 + (o % max(1, vocab_size - 128)))
        return torch.tensor(ids, dtype=torch.long)

    def _encode_condition(self, text_ids: torch.Tensor, emotion_embedding: torch.Tensor) -> torch.Tensor:
        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)

        x = self.text_embedding(text_ids)
        h, _ = self.text_encoder(x)
        text_vec = self.text_proj(h.mean(dim=1))

        emo = emotion_embedding
        if emo.dim() == 1:
            emo = emo.unsqueeze(0)
        emo_vec = self.emo_proj(emo)

        cond = self.fusion(torch.cat([text_vec, emo_vec], dim=-1))
        return cond.squeeze(0)

    def _render_waveform(self, cond: torch.Tensor, text_len: int, device: torch.device) -> torch.Tensor:
        prosody = self.prosody_head(cond)
        harmonics = torch.sigmoid(self.harmonic_head(cond))

        pitch_hz = 95.0 + 185.0 * torch.sigmoid(prosody[0])
        speech_rate = 0.78 + 0.55 * torch.sigmoid(prosody[1])
        energy = 0.12 + 0.55 * torch.sigmoid(prosody[2])
        noise_level = 0.004 + 0.05 * torch.sigmoid(prosody[3])
        vibrato_depth = 0.0 + 0.03 * torch.sigmoid(prosody[4])
        brightness = 0.25 + 0.75 * torch.sigmoid(prosody[5])

        duration_sec = torch.clamp(
            0.85 + (text_len * 0.048) / speech_rate,
            min=0.9,
            max=16.0,
        )
        n_samples = int((duration_sec * self.sample_rate).item())
        t = torch.linspace(0.0, duration_sec.item(), n_samples, device=device)

        vibrato = 1.0 + vibrato_depth * torch.sin(2.0 * torch.pi * 4.8 * t)
        f0 = pitch_hz * vibrato

        phase_base = 2.0 * torch.pi * torch.cumsum(f0 / self.sample_rate, dim=0)
        voice = torch.zeros_like(t)
        for k in range(1, 7):
            amp = harmonics[k - 1] / k
            voice = voice + amp * torch.sin(k * phase_base)

        syllables = max(3, min(30, text_len // 2 + 1))
        env = 0.55 + 0.45 * torch.sin(torch.linspace(0, syllables * torch.pi, n_samples, device=device)).abs()
        env = env.pow(1.15)

        noise = torch.randn(n_samples, device=device)
        noise = 0.75 * noise + 0.25 * torch.roll(noise, shifts=1)
        noise = noise * (1.25 - brightness)

        waveform = energy * env * voice + noise_level * noise
        waveform = torch.tanh(1.8 * waveform)
        waveform = waveform / (waveform.abs().max() + 1e-6)
        return waveform

    def synthesize(self, text: str, emotion_embedding: torch.Tensor) -> torch.Tensor:
        device = emotion_embedding.device
        text_ids = self._text_to_ids(text, self.vocab_size).to(device)
        cond = self._encode_condition(text_ids=text_ids, emotion_embedding=emotion_embedding)
        wav = self._render_waveform(cond=cond, text_len=text_ids.numel(), device=device)
        return wav


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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tts_model = ConditionalNeuralTTSBackbone(sample_rate=SAMPLE_RATE).to(self._device)
        self._lookup_table = EmotionLookupTable().to(self._device)

        if not self._is_stub:
            self._load_model()

    def _load_model(self):
        """Load fine-tuned TTS model and optional lookup table checkpoint."""
        if not self.tts_weights_path:
            raise ValueError("tts_weights_path is empty")
        if not os.path.exists(self.tts_weights_path):
            raise FileNotFoundError(f"Phase3 weights not found: {self.tts_weights_path}")

        ckpt = torch.load(self.tts_weights_path, map_location=self._device)
        if isinstance(ckpt, dict) and "tts_model" in ckpt:
            self._tts_model.load_state_dict(ckpt["tts_model"], strict=True)
            if "lookup_table" in ckpt:
                self._lookup_table.load_state_dict(ckpt["lookup_table"], strict=True)
        elif isinstance(ckpt, dict):
            self._tts_model.load_state_dict(ckpt, strict=False)
        else:
            raise ValueError(
                "Unsupported checkpoint format. "
                "Expected dict with key 'tts_model' or raw state_dict dict."
            )

        self._tts_model.eval()

    @staticmethod
    def _validate_emotion_embedding(emotion_embedding: torch.Tensor):
        if not isinstance(emotion_embedding, torch.Tensor):
            raise TypeError(
                f"emotion_embedding must be torch.Tensor, got {type(emotion_embedding).__name__}"
            )
        if emotion_embedding.shape != (D_EMO,):
            raise ValueError(
                f"Expected embedding shape ({D_EMO},), got {tuple(emotion_embedding.shape)}"
            )

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
        self._validate_emotion_embedding(emotion_embedding)

        text = (text or "").strip()
        if not text:
            duration_sec = 1.0
            n_samples = int(duration_sec * SAMPLE_RATE)
            return np.zeros(n_samples, dtype=np.float32)

        emb = emotion_embedding.to(self._device, dtype=torch.float32)
        with torch.no_grad():
            audio_t = self._tts_model.synthesize(text=text, emotion_embedding=emb)
        return audio_t.detach().cpu().numpy().astype(np.float32)

    def synthesize_from_label(self, text: str, emotion_label: str) -> np.ndarray:
        """
        Synthesize speech directly from discrete emotion label.
        Useful for fallback mode and label-conditioned debugging.
        """
        emb = self._lookup_table.get_embedding_by_label(emotion_label).to(self._device, dtype=torch.float32)
        return self.synthesize_speech(text=text, emotion_embedding=emb)

    def export_lookup_table(self, path: str = LOOKUP_TABLE_PATH):
        """Export current lookup table as JSON (for Phase 2 alignment)."""
        self._lookup_table.export_json(path)

    def save_checkpoint(self, path: str):
        """Save Phase 3 model + lookup table weights for inference/resume."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "tts_model": self._tts_model.state_dict(),
                "lookup_table": self._lookup_table.state_dict(),
                "sample_rate": SAMPLE_RATE,
                "model_name": "ConditionalNeuralTTSBackbone",
            },
            path,
        )
        print(f"Phase3 checkpoint saved to {path}")


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


def synthesize_speech_from_label(text: str, emotion_label: str) -> np.ndarray:
    """Top-level helper: text + emotion label -> waveform."""
    global _default_model
    if _default_model is None:
        _default_model = Phase3Model(tts_weights_path=None)  # stub by default
    return _default_model.synthesize_from_label(text, emotion_label)
