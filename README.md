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

## Phase 3 MVP Workflow (ESD)

This repository now includes a runnable Phase 3 MVP path:

1. **Prepare ESD manifest** (speaker-aware train/val/test split)
2. **Run Phase 3 stub training** (validates data + exports lookup table)
3. **Run inference** (normal path or fallback path)
4. **Run evaluation scaffold** (distribution summary + MOS template)

### 1) Prepare manifest from ESD

```bash
python data/phase3_prepare_data.py --esd-root ESD --output data/phase3_esd_manifest.jsonl
```

### 2) Run Phase 3 stub training

```bash
python data/phase3_train_stub.py \
    --manifest data/phase3_esd_manifest.jsonl \
    --lookup-out weights/emotion_lookup_table.json
```

### 3) Run inference with fallback (Phase 1 label -> Phase 3 lookup table)

```bash
python run_inference.py \
    --image test.jpg \
    --phase1 weights/phase1_lora/ \
    --phase3 weights/phase3_tts.pt \
    --fallback \
    --output outputs/phase3_fallback.wav
```

### 4) Evaluate manifest summary (MVP scaffold)

```bash
python data/phase3_eval_manifest.py \
    --manifest data/phase3_esd_manifest.jsonl \
    --save-json outputs/phase3_manifest_summary.json
```

### Notes

- `ESD/` is local dataset content and should not be committed to git.
- Objective metrics (`MCD`, `CLIP-score`) are tracked as next-step evaluation work.
- Subjective evaluation can start immediately using MOS CSV templates via `create_mos_template` in `emonarrify.phase3.evaluation`.

## System Context & Task Overview:
You are an expert AI software engineer assisting me in implementing an MVP (Minimum Viable Product) for a Multimodal Emotional Voice Generation system, tentatively named "Emotional Image Storytelling Fairy".

To ensure rapid team collaboration and system integration, we are building a simplified engineering version first, utilizing discrete emotion labels and a learnable lookup table rather than continuous feature disentanglement.
Please read the agreed-upon standards, architecture breakdown, datasets, evaluation metrics, and future work below. Acknowledge your understanding, and wait for my instructions to start implementing the stub functions.

1. Agreed-Upon Standards (Strict Constraints)
Emotion Label Set: We strictly use 5 discrete categories aligned with the ESD dataset: ["neutral", "happy", "angry", "sad", "surprise"].
Emotion Embedding Dimension: D_emo = 128. This is used consistently across Phase 2 and Phase 3.
Data Interchange Format: JSON Lines (.jsonl). Every phase must output or consume records with this schema:
{"image_id": str, "image_path": str, "narrative_text": str, "emotion_label": str}
Code Interface Contract: Each phase must expose a single callable function with a fixed signature. We will start by implementing stubs that return dummy data before doing the actual model training.
Example: def generate_story(image: PIL.Image) -> dict:


2. Architecture Breakdown (MVP Version)
PHASE 1: Vision-Language Model (VLM)
Input: Image.
Output: Generates narrative_text and predicts one discrete emotion_label (from the 5 predefined categories).
PHASE 2: Image Emotion Encoder
Input: Image.
Output: Maps the image into a continuous emotion embedding space of size 128 (this vector must eventually align with Phase 3's lookup table).
PHASE 3: Emotion-Conditioned TTS Model
Input: narrative_text and emotion_label.
Core Mechanism: Maintains a learnable Emotion Lookup Table mapping the 5 discrete emotion labels to a 128-dimensional embedding. It uses this lookup table to fetch the exact 128D vector based on the Phase 1 label, and synthesizes the final audio alongside the text.
Export: It exports the trained lookup table (as a JSON mapping discrete labels to 128D floats) for Phase 2 to consume/align with.


3. Data & Evaluation
Datasets for Training:
Vision/Text: VIST (Visual Storytelling Dataset).
Speech/Emotion: Expresso Dataset (preferred), IEMOCAP Emotion Speech Database, EmoV-DB, LibriTTS.
Evaluation Metrics (to be implemented):
Objective: CLIP-score (to evaluate semantic alignment between generated text and input image), MCD / Mel-Cepstral Distortion (to assess acoustic quality and spectral distance against baselines).
Subjective: Mean Opinion Score (MOS) setup for blind tests (evaluating emotional expressiveness and whether the prosody naturally matches the visual atmosphere).


4. [TODO / Future Advanced Upgrades]
Currently, we use a simple Lookup Table. However, our ultimate architectural goal is to implement the CLEAR Module (Cross-Modal De-Redundancy) to handle continuous features.
Future Phase 2: Instead of discrete labels, it will extract dense, continuous Visual Emotion Features ($E_v$) using Contrastive Learning against ground-truth audio.
Future Phase 3 (CLEAR Module): We will remove the lookup table. Instead, we will calculate the Cross-Covariance between Text Features ($E_t$) and Visual/Audio Features ($E_v$/$E_a$), apply SVD, and use a Null-Space Projection to eliminate shared semantic directions. This will pass only "purified" atmospheric cues to the TTS, preventing emotion collision.

End of Context.
If you understand these MVP constraints, the architecture, datasets, evaluation metrics, and the Future Work trajectory, please reply with a brief confirmation. Let me know when you are ready to write the Python stub functions (returning dummy data) for all 3 phases so we can test the End-to-End pipeline immediately.

