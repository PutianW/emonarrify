# EmoNarrify

**Emotional Image-to-Audio Story Generation** — 18-789 Course Project, CMU Spring 2026


## Project Structure

```
emonarrify/                                    # git repo root (NOT a Python package itself)
├── emonarrify/                                # Python package (importable as `from emonarrify import ...`)
│   ├── __init__.py                            # Exports EmoNarrifyPipeline
│   ├── config.py                              # Shared constants (D_EMO=256, EMOTION_LABELS, SAMPLE_RATE=16000, paths)
│   ├── pipeline.py                            # EmoNarrifyPipeline orchestrator (image -> wav)
│   ├── phase1/
│   │   ├── __init__.py
│   │   └── model.py                           # Phase1Model: VLM + LoRA (image -> story + emotion label); stub mode supported
│   ├── phase2/
│   │   ├── __init__.py
│   │   ├── dataset.py                         # Phase2CachedDataset (CLIP feature cache loader)
│   │   └── model.py                           # MLPProjectionHead (CLIP 512d -> D_EMO 256d) + Phase2Model wrapper
│   └── phase3/
│       ├── __init__.py
│       ├── data.py                            # ESD dataset utilities (manifest builder, label normalization)
│       ├── model.py                           # Phase3Model (delegates to PatchedVITSBackbone in deployed mode)
│       └── vits_backbone.py                   # PatchedVITSBackbone: emotion-conditioned VITS inference wrapper
├── vits/                                      # Vendored upstream VITS (jaywalnut310 master) + emotion-conditioning patches
│   ├── models.py                              # SynthesizerTrn with emb_e + classifier head + emotion_embedding_override
│   ├── data_utils.py                          # TextAudioEmotionLoader + Collate (emotion-id batching)
│   ├── mel_processing.py                      # torch 2.x + librosa 0.10+ compatibility patches
│   ├── configs/phase3_new_v1.json             # Phase 3 training config (lambda_cls, lambda_ortho, model arch)
│   └── ...                                    # Other upstream files (text/, monotonic_align/, train_ms.py, etc.)
├── scripts/                                   # Training, evaluation, and demo scripts
│   ├── phase2_train.py                        # Phase 2 MLP training (--loss-reweight, --oversample)
│   ├── phase2_eval.py                         # Phase 2 per-split evaluation
│   ├── phase2_extract_clip.py                 # Pre-compute CLIP ViT-B/32 features
│   ├── phase2_validate_lookup.py              # Sanity-check emotion_lookup_table.json schema
│   ├── phase3_new_train.py                    # Phase 3 VITS training (--lambda-cls, --lambda-ortho, --model-dir overrides)
│   ├── phase3_new_prepare.py                  # Build ESD manifest + train/val/test filelists
│   ├── phase3_new_pre_clean_text.py           # Persistent EspeakBackend pre-cleaning (RSS-leak fix)
│   ├── eval_e2e_ser.py                        # End-to-end SER evaluation (Tasks 1 + 2a, two-pass for env split)
│   ├── eval_phase3_isolated_ser.py            # Phase 3 isolated SER evaluation (Task 2b + ablations)
│   ├── build_demo.py                          # 10-wav demo set builder (2 per emotion)
│   └── build_samples.py                       # 5-sample exhibit builder (one image per emotion)
├── notebooks/
│   ├── phase1_train.ipynb                     # Phase 1 VLM + LoRA training notebook
│   └── phase3_train.ipynb                     # Phase 3 early VITS training notebook
├── data/                                      # Datasets and split metadata (mostly gitignored, ~13 GB)
│   ├── ESD/                                   # Emotional Speech Dataset (Phase 3 audio source, gitignored)
│   ├── images/                                # VIST images (Phase 1 + 2 visual source, gitignored)
│   ├── features/                              # Pre-computed CLIP ViT-B/32 feature caches (gitignored)
│   ├── filelists/                             # Phase 3 raw + cleaned filelists (audio|eid|text)
│   ├── splits/                                # Phase 1 splits (train/val/test/phase1_holdout JSONL)
│   ├── phase3_esd_manifest_new.jsonl          # Phase 3 ESD manifest
│   └── phase1_vist_*.jsonl + predictions_*.jsonl   # Phase 1 splits + prior classifier outputs
├── weights/                                   # Model checkpoints (mostly gitignored, ~2.3 GB on disk)
│   ├── emotion_lookup_table.json              # Phase 3 Run 1 lookup table (committed, ~36 KB)
│   ├── phase2_mlp.pt                          # Phase 2 v3 MLP head (committed, ~1.6 MB)
│   ├── phase1_lora/                           # Phase 1 LoRA adapter weights (gitignored)
│   ├── vits_vctk/pretrained_vctk.pth          # VCTK pretrained init for Phase 3 training (gitignored)
│   ├── phase3_new_v1/G_18000.pth + D_18000.pth     # Run 1 deployed Phase 3 generator + discriminator (gitignored)
│   ├── phase3_new_v2_no_cls/G_14000.pth       # Stage G Run 2 ablation reference (gitignored)
│   └── phase3_new_v3_no_ortho/G_14000.pth     # Stage G Run 3 ablation reference (gitignored)
├── outputs/                                   # Generated artifacts (~6 MB tracked)
│   ├── demo/                                  # 10-wav demo set + manifest.json
│   ├── samples/                               # 5-sample exhibit + manifest.json (one image per emotion)
│   ├── phase3_new_v1_listening/               # Phase 3 Run 1 original 5-emotion listening set
│   ├── phase3_new_v2_listening/               # Phase 3 Run 1 reproduction listening (deployed)
│   ├── phase3_new_v2_no_cls_listening/        # Stage G Run 2 ablation listening (no L_cls)
│   ├── phase3_new_v3_no_ortho_listening/      # Stage G Run 3 ablation listening (no L_ortho)
│   ├── e2e_smoke.wav + e2e_smoke_metadata.json     # Stage F end-to-end pipeline smoke evidence
│   ├── phase3_new_v*_analysis.json            # Per-checkpoint emb_e geometry + drift analysis (4 runs)
│   ├── ser_e2e_eval*.json + pipeline.jsonl    # End-to-end SER evaluation (Task 1 + 2a)
│   ├── ser_phase3_isolated_eval*.json + pipeline.jsonl   # Phase 3 isolated SER (Task 2b + ablations)
│   └── phase2_*v3*.log                        # Phase 2 v3 training and evaluation audit logs
├── requirements.txt                           # Pinned conda env dependencies (Python 3.10)
└── README.md                                  # This file
```

## Listening samples

The repository ships with several listening sets that correspond to claims and evaluations in the report.

**Paper-referenced demo and exhibit sets:**

- `outputs/demo/` -- 10 wav (5 emotions x 2 images each) drawn from the test split. Cited in Section 4.4 of the paper. Each wav synthesizes audio for an image whose ground-truth emotion is recorded in `manifest.json`.
- `outputs/samples/` -- 5 wav (5 emotions x 1 image each) drawn deterministically from the test split, deduped against the demo set. Cited in the Reproducibility appendix.

**Phase 3 ablation listening sets** (Section 4.2):

- `outputs/phase3_new_v1_listening/` -- 5 wav (Run 1 with both auxiliary losses, deployed configuration).
- `outputs/phase3_new_v2_listening/` -- 5 wav (Run 1 reproduction after the storage-failure incident; perceptually equivalent to v1).
- `outputs/phase3_new_v2_no_cls_listening/` -- 5 wav (Auxiliary-loss ablation Run 2, lambda_cls=0; demonstrates the perceptual intensity loss when the classification loss is removed).
- `outputs/phase3_new_v3_no_ortho_listening/` -- 5 wav (Auxiliary-loss ablation Run 3, lambda_ortho=0; demonstrates that the orthogonality fence is dormant in the deployed regime).

**Suggested listening sequence:** start with `outputs/demo/` for general system behavior, then listen to `outputs/samples/` for one-per-emotion intuition, then A/B between `outputs/phase3_new_v2_listening/` (deployed) and `outputs/phase3_new_v2_no_cls_listening/` (no L_cls ablation) on the same emotion to hear the prosodic intensity contribution discussed in the paper.

## Setup

Two weight files are tracked in git directly because they are small:
- `weights/phase2_mlp.pt` (~1.5 MB) -- deployed Phase 2 MLP head
- `weights/emotion_lookup_table.json` (~36 KB) -- Phase 3 Run 1 emotion lookup table

All other model checkpoints (Phase 3 generator/discriminator, Phase 1 LoRA, VCTK initialization) are gitignored and distributed via the Google Drive bundle described in Setup.

### Hardware

We trained and evaluated on AWS EC2 g6e.xlarge with an NVIDIA L40S GPU (48 GB VRAM). Inference runs on smaller GPUs (the deployed pipeline uses approximately 12 GB at 16 kHz output), but Phase 3 training requires ~24 GB for the configuration described below.

### Environment

We tested on Python 3.10. The pinned `requirements.txt` uses NumPy 2.2 and PyTorch 2.11, both of which require Python 3.10 or newer.

```bash
conda create -n emona python=3.10
conda activate emona
pip install -r requirements.txt
```

After installing the Python dependencies, install the phonemizer system dependency and build the VITS monotonic alignment Cython extension:

```bash
sudo apt install espeak-ng
cd vits/monotonic_align && python setup.py build_ext --inplace && cd ../..
```

Both steps are required for Phase 3 inference and training to work; skipping either will produce import-time errors.

For dataset and weight downloads via Google Drive, also install `gdown`:

```bash
pip install gdown
```

### Datasets

Training data is not included in the repository. Re-download as follows.

**ESD (Emotional Speech Dataset):** used for Phase 3 training.

```bash
mkdir -p data/ESD
cd data/ESD
gdown 1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v -O ESD.zip
unzip ESD.zip
cd ../..
```

**VIST (Visual Storytelling Dataset):** used for Phase 1 LoRA training and Phase 2 CLIP feature extraction. Obtain from the official VIST distribution and place under `data/images/`. The jsonl partitions used by our pipeline are tracked at `data/splits/{train,val,test,phase1_holdout}.jsonl`.

### Model weights

Download the deployed and ablation model checkpoints from Google Drive:

> **Weights bundle:** https://drive.google.com/file/d/1699KLzQeecedeCT13TSGmk4krfNTj7V1/view?usp=sharing

```bash
# Download from Google Drive (manual download via browser, or gdown)
gdown 1699KLzQeecedeCT13TSGmk4krfNTj7V1 -O weights_bundle.tar.gz

# Extract into the weights/ directory (does not overwrite the two tracked files)
tar -xzf weights_bundle.tar.gz -C weights/

# Sanity check
ls weights/phase3_new_v1/G_18000.pth      # Run 1 deployed generator (~478 MB)
ls weights/phase2_mlp.pt                  # tracked, should already exist
ls weights/emotion_lookup_table.json      # tracked, should already exist
```

The bundle contains the gitignored checkpoints needed for inference and ablation reproduction:

- `phase3_new_v1/G_18000.pth` + `D_18000.pth` -- Run 1 deployed Phase 3 (~1 GB combined)
- `phase3_new_v2_no_cls/G_14000.pth` -- Auxiliary-loss ablation Run 2 ablation last-saved checkpoint
- `phase3_new_v3_no_ortho/G_14000.pth` -- Auxiliary-loss ablation Run 3 ablation last-saved checkpoint
- `phase1_lora/` -- Cassie's Qwen3-VL LoRA adapter
- `vits_vctk/pretrained_vctk.pth` -- VCTK initialization for Phase 3 retraining

## Inference

Run the deployed pipeline on a single image:

```python
from PIL import Image
from emonarrify import EmoNarrifyPipeline
import soundfile as sf

pipe = EmoNarrifyPipeline(
    phase1_adapter=None,                              # stub mode (see Methodology)
    phase2_mlp='weights/phase2_mlp.pt',
    phase3_tts='weights/phase3_new_v1/G_18000.pth',
)

image = Image.open('path/to/image.jpg').convert('RGB')
result = pipe.run(image)

sf.write('output.wav', result['audio'], result['sample_rate'])
print(f"Audio: {len(result['audio']) / result['sample_rate']:.2f}s")
print(f"Phase 1 emotion: {result.get('emotion_label')}")
print(f"Phase 1 narrative: {result.get('narrative_text')}")
```

The `phase1_adapter=None` argument selects the deterministic stub Phase 1 used for all evaluations reported in the paper. To invoke the LoRA-finetuned Phase 1, pass `phase1_adapter='weights/phase1_lora'` instead. Note that LoRA inference requires additional dependencies not pinned in `requirements.txt` (the install is heavy and CUDA-version-fragile); install them separately:

```bash
pip install unsloth transformers datasets
```

## Training

### Phase 1: LoRA fine-tuning of Qwen3-VL

The training notebook is preserved at `notebooks/phase1_train.ipynb`; refer to it for the full configuration and dataset preparation flow. The deployed adapter is included in the Google Drive weights bundle.

### Phase 2: CLIP-MLP image emotion encoder

```bash
# 1. Precompute CLIP image features (one-time, requires data/images/)
python scripts/phase2_extract_clip.py

# 2. Validate that Phase 3's lookup table has been exported (run after Phase 3)
python scripts/phase2_validate_lookup.py

# 3. Train the MLP head
python scripts/phase2_train.py --oversample
```

### Phase 3: VITS fine-tuning with auxiliary emotion losses

```bash
# 1. Build the ESD training manifest (one-time)
python scripts/phase3_new_prepare.py

# 2. Pre-phonemize text and cache (one-time)
python scripts/phase3_new_pre_clean_text.py

# 3. Train (deployed configuration: lambda_cls=1.0, lambda_ortho=0.05)
python scripts/phase3_new_train.py \
    -c vits/configs/phase3_new_v1.json \
    --max-steps 20000
```

### Auxiliary-loss ablation runs

The training script accepts CLI overrides for the auxiliary loss weights and an alternative model directory, supporting controlled ablation:

```bash
# Run 2: ablate L_cls (set its weight to 0)
python scripts/phase3_new_train.py \
    -c vits/configs/phase3_new_v1.json \
    --lambda-cls 0.0 \
    --max-steps 15000 \
    --model-dir weights/phase3_new_v2_no_cls

# Run 3: ablate L_ortho (set its weight to 0)
python scripts/phase3_new_train.py \
    -c vits/configs/phase3_new_v1.json \
    --lambda-ortho 0.0 \
    --max-steps 15000 \
    --model-dir weights/phase3_new_v3_no_ortho
```

Each Auxiliary-loss ablation run takes approximately 1.7 hours on the L40S. The deployed Run 1 takes approximately 1.84 hours.

## Reproducing paper evaluations

### Phase 2 evaluation (Section 4.1, Table 1)

```bash
python scripts/phase2_eval.py
```

Produces per-split macro accuracy and mean cosine similarity on train, val, test, and Phase 1 holdout splits.

### Phase 3 ablation analysis (Section 4.2, Table 2)

After running the three Auxiliary-loss ablation configurations, the per-checkpoint analysis JSONs (`outputs/phase3_new_v*_analysis.json`) are produced as a side effect of training. The deployed checkpoint selection logic is in `scripts/phase3_new_train.py`.

### SER end-to-end evaluation (Section 4.3, Table 3)

```bash
# Task 1: full pipeline (Phase 2 path)
python scripts/eval_e2e_ser.py

# Task 2a: fallback baseline (lookup neutral instead of Phase 2)
python scripts/eval_e2e_ser.py --fallback

# Task 2b: Phase 3 isolated (5 emotions x 11 narratives, balanced)
python scripts/eval_phase3_isolated_ser.py
```

### Demo set and 5-sample exhibit (Section 4.4 and Reproducibility appendix)

```bash
# 10-utterance demo (5 emotions x 2 images each, from test split)
python scripts/build_demo.py

# 5-utterance one-per-emotion exhibit (deduped against demo)
python scripts/build_samples.py
```

## Citation and acknowledgments

This project was completed for **CMU 18-789 (Deep Generative AI), Spring 2026**, by:

- Andrew Chen
- Cassie Chang
- Putian Wang

If you reference this work, please cite the course project:

```bibtex
@misc{emonarrify2026,
  title  = {EmoNarrify: End-to-End Emotional Story Narration from Images},
  author = {Andrew Chen and Cassie Chang and Putian Wang},
  year   = {2026},
  note   = {Final project report, CMU 18-789 (Deep Generative AI), Spring 2026}
}
```

We acknowledge the following upstream models and datasets:

- **Qwen3-VL-8B** (Bai et al., 2023) -- vision-language backbone for Phase 1.
- **CLIP ViT-B/32** (Radford et al., 2021) -- frozen image encoder for Phase 2.
- **VITS** (Kim et al., 2021) -- text-to-speech backbone for Phase 3.
- **VIST** (Huang et al., 2016) -- image-narrative training corpus.
- **ESD** (Zhou et al., 2021) -- emotional speech corpus.
- **GoEmotions** (Demszky et al., 2020) -- text-emotion auto-labeling.
- **SUPERB wav2vec2-base-superb-er** (Yang et al., 2021) -- SER classifier used for end-to-end audio evaluation.
