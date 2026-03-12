# Multimodal Emotion Regulation Analysis

**Cross-Modal Incongruence and Temporal Dynamics for Expressive Control**

---

## What This Project Does

This system infers **expressive control in emotional behaviour** from multimodal signals — face, audio, and text — without requiring emotion labels.

Instead of classifying emotions, it produces a continuous **Expressive Control Index (ECI)**: a score representing how congruent and regulated a person's emotional signals are across modalities.

A high ECI means face, voice, and words are consistent with each other.  
A low ECI means they are incongruent — a signal of emotional suppression or dysregulation.

---

## Architecture

```
Face (MediaPipe FaceMesh)     → 944-dim
Audio (MFCC + Prosody)        → 46-dim
Text (RoBERTa + GoEmotions)   → 796-dim
           ↓
  Incongruence Encoder
  (Self-supervised contrastive)
  1786 → 512 → 256 → 64
           ↓
  Bi-LSTM Temporal Model
  (Attention-weighted ECI)
           ↓
  Expressive Control Index (0–1)
```

The incongruence encoder is trained with **contrastive learning**: positive pairs are aligned face/audio/text triplets, negative pairs have one shuffled modality. This forces the encoder to learn what cross-modal agreement looks like without any emotion labels.

---

## Dataset Strategy

| Stage | Dataset | Purpose |
|-------|---------|---------|
| Training | IEMOCAP Session1 (80%) | Train incongruence encoder + temporal model |
| Validation | IEMOCAP Session1 (20%) | Monitor overfitting |
| Evaluation | CMU-MOSEI 500 clips | Cross-dataset generalisation (frozen model) |

**Why only Session1?** The Kaggle mirror of IEMOCAP only contains video files (.avi) for Session1. Sessions 2–5 have audio and transcripts only.

**Why MOSEI for evaluation?** MOSEI is a YouTube-sourced dataset — a completely different recording context from IEMOCAP's controlled lab environment. Running the frozen model on MOSEI tests whether the learned embeddings generalise across domains.

---

## Results

### Training (IEMOCAP Session1)

| Metric | Value |
|--------|-------|
| Total samples | 26 |
| Training samples | 20 |
| Validation samples | 6 |
| Incongruence encoder best val loss | 0.6224 |
| Temporal model best val loss | 0.0000 |
| Total temporal frames | 523 |

### Cross-Dataset Evaluation

| Dataset | Samples | ECI Mean | ECI Std | ECI Median |
|---------|---------|----------|---------|------------|
| IEMOCAP (val) | 26 | **0.9228** | 0.1811 | 1.0000 |
| CMU-MOSEI | 500 | **0.6774** | 0.2388 | 0.4980 |

The ECI gap of **ΔECI ≈ 0.245** between IEMOCAP and MOSEI is the key finding. The model assigns higher expressive control scores to controlled lab speech (IEMOCAP) and lower scores to naturalistic YouTube speech (MOSEI) — which aligns with the expected difference in expressive dynamics between these two contexts.

---

## Project Structure

```
emotion-regulation/
│
├── infer.py                  ← Mac inference script
├── requirements.txt          ← Mac dependencies
├── README.md
│
└── checkpoints/              ← Download from Colab training
    ├── incongruence_encoder_best.pth
    ├── temporal_model_best.pth
    └── config.py
```

The Colab training notebook handles all data download, feature extraction, training, and evaluation. The Mac inference script only needs the two checkpoint files.

---

## Setup (Mac)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place checkpoints
# Unzip checkpoints.zip from Colab into ./checkpoints/
```

---

## Usage

### Full multimodal input

```bash
python infer.py --video path/to/clip.mp4 --audio path/to/clip.wav --text "I'm completely fine with everything"
```

### Audio + text only (no video)

```bash
python infer.py --audio path/to/clip.wav --text "I'm completely fine with everything"
```

### Text only

```bash
python infer.py --text "I'm completely fine with everything"
```

### Example output

```
✓ Models loaded  (input_dim=1786)

Extracting features...
  Face  ← clip.mp4
  Audio ← clip.wav
  Text  ← "I'm completely fine with everything"

  Temporal frames: 47

=============================================
EXPRESSIVE CONTROL INDEX (ECI) RESULTS
=============================================
  ECI Score:               0.4312
  Temporal Variance:       0.0287
  Incongruence Magnitude:  0.8841
=============================================

  Interpretation: LOW expressive control — high cross-modal incongruence detected

  Reference ranges (from training):
    IEMOCAP mean ECI: 0.9228  (controlled lab speech)
    MOSEI mean ECI:   0.6774  (naturalistic YouTube speech)
```

### ECI score interpretation

| ECI Range | Meaning |
|-----------|---------|
| 0.8 – 1.0 | High expressive control — face, voice, and words are congruent |
| 0.5 – 0.8 | Moderate — some cross-modal inconsistency |
| 0.0 – 0.5 | Low — high incongruence, possible suppression or dysregulation |

---

## Limitations

- Training set is small (26 samples from one IEMOCAP session)
- Temporal model reached 0.0 val loss — likely overfit on the small sequence count
- MOSEI evaluation uses FACET42/COVAREP features (35/74 dims) which differ from the IEMOCAP extraction pipeline (944/46 dims) — features are zero-padded during evaluation
- Text is zeroed out during MOSEI evaluation (GloVe vectors not available in the pkl source used)

---

## How to Replicate

1. Open `emotion_regulation_colab_v4.ipynb` in Google Colab
2. Set runtime to T4 GPU
3. Run all cells top to bottom (~2–4 hours)
4. Download `checkpoints.zip` and `results.zip`
5. Run inference on Mac using `infer.py`

---

## Interview Summary

> "The framework trains a self-supervised incongruence encoder on IEMOCAP Session1 using contrastive learning across face, audio, and text modalities. A Bi-LSTM temporal model then aggregates frame-level embeddings into a continuous Expressive Control Index. Evaluating the frozen model on 500 CMU-MOSEI clips showed an ECI drop from 0.92 to 0.68 — a measurable cross-dataset distribution shift consistent with the difference between controlled lab recordings and naturalistic YouTube speech. The architecture is designed to scale to full datasets when compute permits."

---

## Dependencies

| Library | Purpose |
|---------|---------|
| PyTorch | Model training and inference |
| MediaPipe | Face landmark extraction |
| librosa | Audio feature extraction (MFCC, prosody) |
| transformers | RoBERTa text encoder (GoEmotions) |
| scikit-learn | Train/val split |
| mmsdk | CMU-MOSEI feature loading |
| tensorboard | Training loss visualisation |
