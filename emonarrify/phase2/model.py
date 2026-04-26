"""
Phase 2: Image Emotion Encoder
Owner: Member C

Interface contract:
    encode_image_emotion(image: PIL.Image) -> torch.Tensor
    Returns emotion embedding, shape (D_emo,)

Dependencies:
    - Phase 3: trained emotion lookup table (JSON)
    - Phase 1: training set .jsonl with (image_path, emotion_label) pairs
"""

import json
import torch
import torch.nn as nn
from PIL import Image

from ..config import D_EMO, LOOKUP_TABLE_PATH, EMOTION_TO_IDX


class MLPProjectionHead(nn.Module):
    """Lightweight MLP: D_clip -> 512 -> D_emo"""

    def __init__(self, d_clip: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_clip, 512),
            nn.ReLU(),
            nn.Linear(512, D_EMO),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Phase2Model:
    """Image Emotion Encoder: image -> emotion embedding in Phase 3's space."""

    def __init__(self, mlp_weights_path: str = None, clip_model_name: str = "ViT-B/32"):
        """
        Args:
            mlp_weights_path: path to trained MLP projection head weights.
                              If None, runs in stub mode.
            clip_model_name:  CLIP model variant for feature extraction.
        """
        self.mlp_weights_path = mlp_weights_path
        self.clip_model_name = clip_model_name
        self._is_stub = mlp_weights_path is None
        self._clip_model = None
        self._clip_preprocess = None
        self._mlp = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self._is_stub:
            self._load_models()

    def _load_models(self):
        """Load frozen CLIP ViT + trained MLP head."""
        import clip  # lazy: openai-clip pkg
        self._clip_model, self._clip_preprocess = clip.load(
            self.clip_model_name,
            device=self._device,
            jit=False,
        )
        self._clip_model.eval()
        for p in self._clip_model.parameters():
            p.requires_grad = False

        ckpt = torch.load(
            self.mlp_weights_path,
            map_location=self._device,
            weights_only=False,
        )
        d_clip = ckpt.get("d_clip", 512)
        self._mlp = MLPProjectionHead(d_clip=d_clip).to(self._device)
        self._mlp.load_state_dict(ckpt["state_dict"])
        self._mlp.eval()

    def encode_image_emotion(self, image: Image.Image) -> torch.Tensor:
        """
        Map an image to a continuous emotion embedding in Phase 3's space.

        Args:
            image: PIL.Image input

        Returns:
            torch.Tensor of shape (D_emo,)
        """
        if self._is_stub:
            return torch.zeros(D_EMO)

        preprocessed = self._clip_preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            clip_feature = self._clip_model.encode_image(preprocessed).float()
            embedding = self._mlp(clip_feature)
        return embedding.squeeze(0).cpu()

    @staticmethod
    def load_lookup_table(path: str = LOOKUP_TABLE_PATH) -> dict:
        """
        Load Phase 3's exported emotion lookup table.
        Utility for training and for the fallback mechanism.

        Returns:
            dict mapping emotion_label -> torch.Tensor of shape (D_emo,)
        """
        with open(path, "r") as f:
            data = json.load(f)

        assert data["embedding_dim"] == D_EMO, (
            f"Lookup table dim {data['embedding_dim']} != config D_EMO {D_EMO}"
        )

        table = {}
        for label, vec in data["embeddings"].items():
            table[label] = torch.tensor(vec, dtype=torch.float32)
        return table


# =========================================================================
# Convenience function (matches interface contract)
# =========================================================================
_default_model: Phase2Model = None


def encode_image_emotion(image: Image.Image) -> torch.Tensor:
    """Top-level interface function as specified in the project contract."""
    global _default_model
    if _default_model is None:
        _default_model = Phase2Model(mlp_weights_path=None)  # stub by default
    return _default_model.encode_image_emotion(image)
