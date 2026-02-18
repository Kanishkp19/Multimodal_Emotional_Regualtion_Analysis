# Multimodal Emotion Regulation Analysis

## Project Overview

This project presents an **offline multimodal analysis framework** for studying **expressive control in human emotional behavior**, using **cross-modal incongruence and temporal stability** as proxy signals.

### Core Research Question
> *Can expressive control (emotional self-regulation signals) be inferred from cross-modal incongruence and temporal dampening patterns in multimodal speech?*

### Key Features
- ✅ Cross-modal incongruence learning (face, voice, text)
- ✅ Temporal dynamics modeling with Bi-LSTM
- ✅ Self-supervised training approach
- ✅ Expressive Control Index (ECI) output
- ✅ Dataset separation: IEMOCAP (train) + CMU-MOSEI (validate)

---

## System Requirements

### Hardware
- **Processor**: Apple M4 chip (or any modern CPU)
- **RAM**: 16 GB
- **Storage**: ~40-60 GB for datasets
- **GPU**: Not required (CPU training is supported)

### Software
- **OS**: macOS (optimized for M4)
- **Python**: 3.9+
- **pip**: Latest version

---

## Installation Guide

### Step 1: Clone/Setup Repository
```bash
cd /path/to/emotion-regulation-analysis
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Setup Kaggle API (for IEMOCAP)
```bash
# Place your kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
# Download kaggle.json from https://www.kaggle.com/settings
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Dataset Setup

### IEMOCAP Dataset
```bash
python preprocessing/download_iemocap.py
```

### CMU-MOSEI Dataset
```bash
python preprocessing/download_mosei.py
```

**Storage Note**: Datasets will be downloaded to `data/` directory. Ensure ~50GB free space.

---

## Project Structure

```
emotion-regulation-analysis/
├── data/
│   ├── iemocap/          # IEMOCAP dataset
│   └── mosei/            # CMU-MOSEI dataset
├── preprocessing/
│   ├── download_iemocap.py
│   ├── download_mosei.py
│   ├── extract_features.py
│   └── create_dataset.py
├── encoders/
│   ├── face_encoder.py    # MediaPipe face features
│   ├── audio_encoder.py   # MFCC + prosody
│   └── text_encoder.py    # RoBERTa emotion
├── models/
│   ├── incongruence_encoder.py  # Cross-modal learning
│   └── temporal_model.py        # Bi-LSTM aggregation
├── training/
│   ├── train_incongruence.py
│   └── train_temporal.py
├── analysis/
│   ├── ablation_study.ipynb
│   └── visualization.ipynb
├── utils/
│   ├── config.py
│   └── metrics.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## Usage Pipeline

### Step 1: Extract Features from Raw Data
```bash
# Extract face, audio, and text features
python preprocessing/extract_features.py --dataset iemocap
python preprocessing/extract_features.py --dataset mosei
```

### Step 2: Create Processed Datasets
```bash
# Create aligned multimodal samples
python preprocessing/create_dataset.py
```

### Step 3: Train Incongruence Encoder
```bash
# Self-supervised learning on IEMOCAP
python training/train_incongruence.py \
    --epochs 15 \
    --batch_size 32 \
    --lr 0.001
```

### Step 4: Train Temporal Model
```bash
# Train Bi-LSTM on incongruence sequences
python training/train_temporal.py \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001
```

### Step 5: Evaluate on MOSEI
```bash
# Freeze models and validate on MOSEI
python evaluate.py --dataset mosei
```

### Step 6: Run Analysis
```bash
# Open Jupyter notebooks for analysis
jupyter notebook analysis/ablation_study.ipynb
jupyter notebook analysis/visualization.ipynb
```

---

## Expected Outputs

### Trained Models
- `checkpoints/incongruence_encoder.pth`
- `checkpoints/temporal_model.pth`

### Evaluation Results
- `results/eci_scores_iemocap.csv`
- `results/eci_scores_mosei.csv`
- `results/ablation_results.json`

### Visualizations
- Temporal dynamics plots
- Cross-modal incongruence heatmaps
- ECI distribution comparisons

---

## Key Design Decisions

### 1. Frozen Encoders
All modality encoders are **frozen** (not trained):
- **Face**: MediaPipe landmarks
- **Audio**: MFCC + pitch + energy
- **Text**: RoBERTa (GoEmotions)

### 2. Self-Supervised Learning
Incongruence encoder trained via contrastive learning:
- Positive: aligned multimodal samples
- Negative: shuffled modalities

### 3. Temporal Modeling
Bi-LSTM captures regulation patterns over time windows (10-30 timesteps).

### 4. No Emotion Classification
This project **does not** predict emotions. It models **expressive control** as a continuous signal.

---

## Evaluation Strategy

**No accuracy metrics are used.** Evaluation is based on:
1. **Temporal ablation** (with/without LSTM)
2. **Modality ablation** (removing face/audio/text)
3. **Contrast analysis** (same emotion, different expressiveness)
4. **Cross-dataset validation** (IEMOCAP → MOSEI)

---

## Citation

If you use this code, please cite:

```
@misc{emotion-regulation-analysis,
  title={Multimodal Emotion Regulation via Cross-Modal Incongruence and Temporal Dynamics},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/emotion-regulation-analysis}}
}
```

---

## Troubleshooting

### Issue: Kaggle API not working
**Solution**: Ensure `kaggle.json` is in `~/.kaggle/` with 600 permissions

### Issue: Memory errors during feature extraction
**Solution**: Reduce batch size or process in smaller chunks

### Issue: Slow processing on M4
**Solution**: Use `--num_workers 4` for DataLoader parallelization

---

## License

MIT License - See LICENSE file for details

---

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com]