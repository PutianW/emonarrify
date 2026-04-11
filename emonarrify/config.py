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
D_EMO = 128

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
