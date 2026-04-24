"""
EmoNarrify - Shared Configuration
All cross-phase constants are defined here. Any change must be agreed upon by all members.
"""

import os

# =============================================================================
# Emotion Standards (locked at Checkpoint 1)
# =============================================================================
EMOTION_LABELS = ["neutral", "happy", "angry", "sad", "surprise"]
NUM_EMOTIONS = len(EMOTION_LABELS)
EMOTION_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
IDX_TO_EMOTION = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

# =============================================================================
# Embedding Dimension (locked at Checkpoint 1)
# =============================================================================
D_EMO = 256

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Phase 3 -> Phase 2 handoff
LOOKUP_TABLE_PATH = os.path.join(WEIGHTS_DIR, "emotion_lookup_table.json")

# Phase 1 -> Phase 2 handoff
PHASE1_JSONL_PATH = os.path.join(DATA_DIR, "phase1_train.jsonl")

# =============================================================================
# Audio
# =============================================================================
SAMPLE_RATE = 22050

# =============================================================================
# Default Fallback
# =============================================================================
DEFAULT_EMOTION = "neutral"

# =============================================================================
# Data Interchange Formats
#
# All cross-phase data files must follow the schemas below.
# Do NOT change these without all three members agreeing.
# =============================================================================
#
# --- Phase 1 output: PHASE1_JSONL_PATH (.jsonl) ---
# One JSON object per line. Phase 2 consumes this for training.
#
#   {
#     "image_id": "VIST_00001",
#     "image_path": "data/images/00001.jpg",
#     "narrative_text": "The sky turned dark...",
#     "emotion_label": "sad"                        # must be in EMOTION_LABELS
#   }
#
# --- Phase 3 output: LOOKUP_TABLE_PATH (.json) ---
# Exported after Phase 3 training. Phase 2 uses these vectors as regression targets.
#
#   {
#     "embedding_dim": 256,                         # must equal D_EMO
#     "embeddings": {
#       "neutral":  [0.12, -0.03, ...],             # list of 256 floats
#       "happy":    [0.45,  0.21, ...],
#       "angry":    [...],
#       "sad":      [...],
#       "surprise": [...]
#     }
#   }
#
# --- Function signatures (interface contract) ---
#
#   Phase 1:  generate_story(image: PIL.Image) -> dict
#             returns {"narrative_text": str, "emotion_label": str, "parse_confidence": str}
#
#   Phase 2:  encode_image_emotion(image: PIL.Image) -> torch.Tensor
#             returns emotion embedding, shape (D_EMO,)
#
#   Phase 3:  synthesize_speech(text: str, emotion_embedding: torch.Tensor) -> np.ndarray
#             returns audio waveform, shape (n_samples,)
