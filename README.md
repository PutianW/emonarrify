# EmoNarrify

**Emotional Image-to-Audio Story Generation** — 18-789 Course Project, CMU

## Quick Start

```bash
git clone https://github.com/YOUR_REPO/emonarrify.git
cd emonarrify
pip install -r requirements.txt

# Run with stubs (verify pipeline structure)
python run_inference.py --image test.jpg --output outputs/test.wav

# Run with trained models
python run_inference.py \
    --image test.jpg \
    --phase1 weights/phase1_lora/ \
    --phase2 weights/phase2_mlp.pt \
    --phase3 weights/phase3_tts.pt \
    --output outputs/result.wav

# Fallback mode (skip Phase 2)
python run_inference.py --image test.jpg --phase1 weights/phase1_lora/ --phase3 weights/phase3_tts.pt --fallback
```

## Project Structure

```
emonarrify/                    # git repo root (NOT a Python package)
├── emonarrify/                # Python package
│   ├── __init__.py
│   ├── config.py              # Shared constants (D_EMO=128, labels, paths)
│   ├── pipeline.py            # EmoNarrifyPipeline hyper class
│   ├── phase1/
│   │   ├── __init__.py
│   │   └── model.py           # VLM: image -> story + emotion label
│   ├── phase2/
│   │   ├── __init__.py
│   │   └── model.py           # Image Emotion Encoder: image -> embedding
│   └── phase3/
│       ├── __init__.py
│       └── model.py           # Emotion-conditioned TTS: text + embedding -> audio
├── run_inference.py           # CLI entry point
├── notebooks/
│   ├── phase1_train.ipynb     # Member B training notebook
│   ├── phase2_train.ipynb     # Member C training notebook
│   ├── phase3_train.ipynb     # Member A training notebook
│   └── demo.ipynb             # End-to-end demo
├── data/                      # Datasets & .jsonl exports
├── weights/                   # Model weights & lookup table
├── outputs/                   # Generated audio files
└── requirements.txt
```

## Standards

| Item | Value |
|------|-------|
| Emotion labels | `neutral, happy, angry, sad, surprise` |
| Embedding dim | `D_emo = 128` |
| Data format | JSON Lines (`.jsonl`) |
| Audio sample rate | 22050 Hz |

## Training Order

1. **Phase 3** (Member A) — TTS fine-tuning → exports `emotion_lookup_table.json`
2. **Phase 1** (Member B) — VLM LoRA tuning → exports `phase1_train.jsonl`
3. **Phase 2** (Member C) — Image Emotion Encoder → trained MLP weights

Phase 1 and 3 train in parallel. Phase 2 starts after both complete.

## For Colab Users

In the first cell of any notebook:
```python
!git clone https://github.com/YOUR_REPO/emonarrify.git
%cd emonarrify
!pip install -r requirements.txt
```
Then import and use the project modules directly. **Do not write model logic in the notebook** — keep it in the `.py` files so version control stays clean.
