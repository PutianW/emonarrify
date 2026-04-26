"""Patched VITS inference backbone for emotion-conditioned synthesis.

Wraps a fine-tuned VITS SynthesizerTrn (Phase 3 Run 2 G_18000) plus the
exported emotion_lookup_table.json into a single class that exposes two
inference entry points:

  - synthesize_with_embedding(text, emb): caller supplies a 256-d emotion
    vector (the path used by the deployed pipeline, where Phase 2's MLP
    output is fed via emb_e_override at infer time)
  - synthesize_with_label(text, label): caller supplies a discrete
    emotion label, which is dereferenced through the lookup table

Both paths share the same text -> phoneme -> sequence pipeline that
mirrors the english_cleaners2 cleaner used during training, with a
single persistent EspeakBackend instance so the espeak-ng subprocess is
not re-spawned per call (the function-level phonemize() pattern leaks
~26GB RSS over a few thousand utterances; see
scripts/phase3_new_pre_clean_text.py for the same fix in pre-cleaning).

Inference scales hardcode the values used to generate the listening
samples accepted in commit c55405e (Run 2 Path A wavs):
    noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_VITS_DIR = _REPO_ROOT / "vits"

# vits/ is a vendored upstream tree, not a pip-installable package; inject
# it on sys.path once at module import time so the imports below resolve.
# Path injection is program-lifetime (no try/finally cleanup needed).
if str(_VITS_DIR) not in sys.path:
    sys.path.insert(0, str(_VITS_DIR))

import utils as vits_utils  # noqa: E402  (vits/utils.py)
from models import SynthesizerTrn  # noqa: E402  (vits/models.py)
from text import cleaned_text_to_sequence  # noqa: E402  (vits/text/__init__.py)
from text.cleaners import (  # noqa: E402  (vits/text/cleaners.py)
    convert_to_ascii,
    lowercase,
    expand_abbreviations,
    collapse_whitespace,
)
from phonemizer.backend import EspeakBackend  # noqa: E402

DEFAULT_VITS_CONFIG = str(_REPO_ROOT / "vits/configs/phase3_new_v1.json")
DEFAULT_LOOKUP_TABLE = str(_REPO_ROOT / "weights/emotion_lookup_table.json")
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_NOISE_SCALE_W = 0.8
DEFAULT_LENGTH_SCALE = 1.0


class PatchedVITSBackbone:
    """Emotion-conditioned VITS inference (Phase 3 Run 2 architecture)."""

    def __init__(
        self,
        ckpt_path: str,
        vits_config_path: str = DEFAULT_VITS_CONFIG,
        lookup_table_path: str = DEFAULT_LOOKUP_TABLE,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.hps = vits_utils.get_hparams_from_file(vits_config_path)
        self.sampling_rate = self.hps.data.sampling_rate
        self.add_blank = self.hps.data.add_blank

        # Build SynthesizerTrn with the same arg order used by
        # scripts/phase3_new_train.py so the loaded state_dict aligns.
        from text.symbols import symbols  # noqa: E402  (lazy: depends on sys.path)
        self._symbols = symbols
        self._symbol_to_id = {s: i for i, s in enumerate(symbols)}

        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            n_emotions=self.hps.data.n_emotions,
            **self.hps.model,
        ).to(self.device)
        self.net_g.eval()
        vits_utils.load_checkpoint(ckpt_path, self.net_g, None)

        # Persistent espeak-ng backend; reused across all phonemize() calls
        # to avoid the per-call subprocess churn that leaks RSS heavily.
        self._espeak = EspeakBackend(
            language="en-us",
            preserve_punctuation=True,
            with_stress=True,
        )

        with open(lookup_table_path) as f:
            raw = json.load(f)
        if raw["embedding_dim"] != self.hps.model.gin_channels:
            raise ValueError(
                f"lookup table embedding_dim {raw['embedding_dim']} != "
                f"hps.model.gin_channels {self.hps.model.gin_channels}"
            )
        self.lookup_dict = {
            label: torch.tensor(vec, dtype=torch.float32, device=self.device)
            for label, vec in raw["embeddings"].items()
        }

    def _english_cleaners2(self, text: str) -> str:
        """Inline mirror of vits/text/cleaners.py english_cleaners2 using
        the persistent EspeakBackend instance."""
        text = convert_to_ascii(text)
        text = lowercase(text)
        text = expand_abbreviations(text)
        phonemes = self._espeak.phonemize([text], strip=True)
        return collapse_whitespace(phonemes[0])

    def _text_to_ids(self, text: str) -> torch.LongTensor:
        cleaned = self._english_cleaners2(text)
        ids = cleaned_text_to_sequence(cleaned)
        if self.add_blank:
            blanked = [0] * (2 * len(ids) + 1)
            blanked[1::2] = ids
            ids = blanked
        return torch.LongTensor(ids)

    @torch.no_grad()
    def synthesize_with_embedding(
        self,
        text: str,
        emotion_embedding: torch.Tensor,
    ) -> np.ndarray:
        if emotion_embedding.shape[-1] != self.hps.model.gin_channels:
            raise ValueError(
                f"emotion_embedding last dim {emotion_embedding.shape[-1]} != "
                f"gin_channels {self.hps.model.gin_channels}"
            )

        x_ids = self._text_to_ids(text).to(self.device)
        x = x_ids.unsqueeze(0)
        x_lengths = torch.LongTensor([x_ids.size(0)]).to(self.device)

        emb = emotion_embedding.to(device=self.device, dtype=torch.float32)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)  # (1, 256); infer adds the trailing dim

        audio_t, _, _, _ = self.net_g.infer(
            x, x_lengths,
            emotion_embedding_override=emb,
            noise_scale=DEFAULT_NOISE_SCALE,
            noise_scale_w=DEFAULT_NOISE_SCALE_W,
            length_scale=DEFAULT_LENGTH_SCALE,
        )
        return audio_t.squeeze().cpu().numpy().astype(np.float32)

    def synthesize_with_label(self, text: str, emotion_label: str) -> np.ndarray:
        emb = self.lookup_dict[emotion_label]
        return self.synthesize_with_embedding(text, emb)


if __name__ == "__main__":
    backbone = PatchedVITSBackbone(
        ckpt_path=str(_REPO_ROOT / "weights/phase3_new_v1/G_18000.pth"),
    )
    print(f"[smoke] device={backbone.device} sr={backbone.sampling_rate} "
          f"add_blank={backbone.add_blank} symbols={len(backbone._symbols)} "
          f"lookup_keys={sorted(backbone.lookup_dict.keys())}")

    audio_a = backbone.synthesize_with_label("Hello world.", "happy")
    print(f"[smoke] Path A (label=happy): shape={audio_a.shape} "
          f"dtype={audio_a.dtype} duration={len(audio_a)/backbone.sampling_rate:.2f}s "
          f"peak={float(np.abs(audio_a).max()):.3f} "
          f"rms={float((audio_a**2).mean()**0.5):.4f} "
          f"finite={bool(np.isfinite(audio_a).all())}")

    custom_emb = torch.randn(256)
    audio_b = backbone.synthesize_with_embedding("Hello world.", custom_emb)
    print(f"[smoke] Path B (random emb): shape={audio_b.shape} "
          f"dtype={audio_b.dtype} duration={len(audio_b)/backbone.sampling_rate:.2f}s "
          f"peak={float(np.abs(audio_b).max()):.3f} "
          f"rms={float((audio_b**2).mean()**0.5):.4f} "
          f"finite={bool(np.isfinite(audio_b).all())}")

    print("[smoke] PatchedVITSBackbone smoke PASS")
