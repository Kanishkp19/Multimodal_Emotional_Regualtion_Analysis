# Project Structure

Complete directory structure and file descriptions for the Emotion Regulation Analysis project.

```
emotion-regulation-analysis/
│
├── README.md                          # Main project documentation
├── IMPLEMENTATION_GUIDE.md            # Step-by-step implementation guide
├── PROJECT_STRUCTURE.md               # This file
├── requirements.txt                   # Python dependencies
├── run_pipeline.sh                    # Automated pipeline script
│
├── data/                              # Data directory (created during setup)
│   ├── iemocap/                      # IEMOCAP dataset (downloaded)
│   ├── mosei/                        # CMU-MOSEI dataset (downloaded)
│   └── processed/                    # Processed features (generated)
│       ├── iemocap_features.pkl      # Extracted features
│       ├── iemocap_train.pkl         # Training split
│       ├── iemocap_val.pkl           # Validation split
│       ├── iemocap_temporal.pkl      # Temporal sequences
│       ├── mosei_features.pkl        # Extracted features
│       └── mosei_test.pkl            # Test split
│
├── preprocessing/                     # Data preprocessing scripts
│   ├── __init__.py
│   ├── download_iemocap.py           # Download IEMOCAP via Kaggle API
│   ├── download_mosei.py             # Download CMU-MOSEI via SDK
│   ├── extract_features.py           # Extract multimodal features
│   └── create_dataset.py             # Create train/val/test splits
│
├── encoders/                          # Frozen feature encoders
│   ├── __init__.py
│   ├── face_encoder.py               # MediaPipe face landmarks
│   ├── audio_encoder.py              # MFCC + prosody features
│   └── text_encoder.py               # RoBERTa + GoEmotions
│
├── models/                            # Neural network models
│   ├── __init__.py
│   ├── incongruence_encoder.py       # Cross-modal incongruence learning
│   └── temporal_model.py             # Bi-LSTM temporal aggregation
│
├── training/                          # Training scripts
│   ├── __init__.py
│   ├── train_incongruence.py         # Train incongruence encoder
│   └── train_temporal.py             # Train temporal model
│
├── utils/                             # Utility functions
│   ├── __init__.py
│   ├── config.py                     # Global configuration
│   └── metrics.py                    # Evaluation metrics & utilities
│
├── analysis/                          # Analysis notebooks
│   ├── visualization.ipynb           # ECI visualization & comparison
│   └── ablation_study.ipynb          # Ablation experiments
│
├── checkpoints/                       # Trained model checkpoints (generated)
│   ├── incongruence_encoder_best.pth
│   ├── temporal_model_best.pth
│   └── [epoch checkpoints...]
│
├── logs/                              # TensorBoard logs (generated)
│   ├── incongruence_encoder/
│   └── temporal_model/
│
├── results/                           # Evaluation results (generated)
│   ├── eci_scores_iemocap.csv
│   ├── eci_scores_mosei.csv
│   ├── cross_dataset_comparison.json
│   └── [visualization plots...]
│
└── evaluate.py                        # Main evaluation script

```

---

## File Descriptions

### Root Files

- **README.md**: Comprehensive project overview, research question, architecture, and usage
- **IMPLEMENTATION_GUIDE.md**: Step-by-step commands for complete setup and execution
- **PROJECT_STRUCTURE.md**: This file - complete directory structure documentation
- **requirements.txt**: Python package dependencies
- **run_pipeline.sh**: Automated bash script to run entire pipeline
- **evaluate.py**: Final evaluation script that computes ECI scores

### preprocessing/

Scripts for downloading and preprocessing raw datasets:

- **download_iemocap.py**: Downloads IEMOCAP from Kaggle using API credentials
- **download_mosei.py**: Downloads CMU-MOSEI using CMU Multimodal SDK
- **extract_features.py**: Extracts face, audio, and text features from raw data
- **create_dataset.py**: Creates aligned train/val/test splits with proper formatting

### encoders/

Frozen feature extraction modules (no training):

- **face_encoder.py**: 
  - Uses MediaPipe Face Mesh
  - Extracts 468 facial landmarks (x, y coordinates)
  - Computes expressiveness features (eye openness, mouth movement, etc.)
  - Output: (T, 944) per video

- **audio_encoder.py**:
  - Uses Librosa for signal processing
  - Extracts MFCC (13), delta MFCC (13), delta-delta MFCC (13)
  - Extracts pitch (F0), energy (RMS), ZCR, spectral features
  - Output: (T, 42) per audio

- **text_encoder.py**:
  - Uses RoBERTa fine-tuned on GoEmotions
  - Extracts sentence embeddings (768-dim)
  - Computes emotion logits (28 emotions)
  - Output: (796,) per utterance

### models/

Trainable neural network architectures:

- **incongruence_encoder.py**:
  - Multi-layer MLP for cross-modal learning
  - Input: Concatenated [face, audio, text]
  - Trained via contrastive learning (triplet loss)
  - Learns to detect misalignment between modalities
  - Output: (64,) incongruence embedding

- **temporal_model.py**:
  - Bi-LSTM for temporal aggregation
  - Attention mechanism for weighted pooling
  - Processes sequences of incongruence vectors
  - Output: Scalar ECI score ∈ [0, 1]

### training/

Model training orchestration:

- **train_incongruence.py**:
  - Loads IEMOCAP train/val data
  - Trains incongruence encoder via self-supervised learning
  - Creates positive (aligned) and negative (shuffled) pairs
  - Saves checkpoints to `checkpoints/`
  - Logs metrics to TensorBoard

- **train_temporal.py**:
  - Loads pre-trained incongruence encoder (frozen)
  - Extracts incongruence features from temporal sequences
  - Trains Bi-LSTM to aggregate temporal patterns
  - Predicts ECI from sequence stability
  - Saves checkpoints to `checkpoints/`

### utils/

Shared utilities and configuration:

- **config.py**:
  - All hyperparameters and paths
  - Model architecture configs
  - Training settings
  - Device configuration (CPU for M4)
  - Dataset URLs and paths

- **metrics.py**:
  - Cross-modal distance computation
  - Temporal variance and dampening metrics
  - Contrastive loss functions
  - ECI statistics computation
  - Distribution comparison tests
  - Visualization helpers

### analysis/

Jupyter notebooks for exploration:

- **visualization.ipynb**:
  - Load and visualize ECI scores
  - Compare IEMOCAP vs MOSEI distributions
  - Statistical significance tests
  - Generate publication-quality plots

- **ablation_study.ipynb**:
  - Modality ablation (remove face/audio/text)
  - Temporal ablation (with/without LSTM)
  - Sequence length sensitivity
  - Quantitative ablation results

---

## Data Flow

```
Raw Data (video, audio, text)
    ↓
[preprocessing/extract_features.py]
    ↓
Extracted Features (.pkl files)
    ↓
[preprocessing/create_dataset.py]
    ↓
Processed Datasets (train/val/test)
    ↓
[training/train_incongruence.py]
    ↓
Trained Incongruence Encoder (.pth)
    ↓
[training/train_temporal.py]
    ↓
Trained Temporal Model (.pth)
    ↓
[evaluate.py]
    ↓
ECI Scores & Analysis (.csv, .json)
    ↓
[analysis/*.ipynb]
    ↓
Visualizations & Insights
```

---

## Key Design Patterns

### 1. Frozen Encoders Pattern
All feature extractors are pre-trained and frozen:
- No backpropagation through encoders
- Ensures reproducibility
- Faster training
- Better generalization

### 2. Self-Supervised Learning
Incongruence encoder trained without labels:
- Positive pairs: aligned modalities
- Negative pairs: shuffled modalities
- Learns structure, not emotion categories

### 3. Dataset Separation
Strict train/test separation:
- IEMOCAP: Training + validation
- MOSEI: Testing only (zero-shot)
- Never mix datasets during training

### 4. Modular Architecture
Each component is independent:
- Encoders can be swapped
- Models can be trained separately
- Easy to extend or modify

---

## File Sizes (Approximate)

| File/Directory | Size |
|----------------|------|
| IEMOCAP (raw) | ~12 GB |
| MOSEI (raw) | ~15-20 GB |
| Processed features | ~500 MB - 1 GB |
| Model checkpoints | ~50-100 MB |
| Results & logs | ~10-50 MB |
| **Total** | **~30-35 GB** |

---

## Dependencies Tree

```
Core:
├── torch (deep learning)
├── numpy (numerical computing)
└── pandas (data manipulation)

Media Processing:
├── librosa (audio)
├── opencv-python (video)
├── mediapipe (face detection)
└── soundfile (audio I/O)

NLP:
├── transformers (RoBERTa)
└── tokenizers (text processing)

Analysis:
├── matplotlib (plotting)
├── seaborn (statistical viz)
└── scikit-learn (metrics)

Data:
├── kaggle (dataset download)
└── CMU-MultimodalSDK (MOSEI)

Logging:
└── tensorboard (training viz)
```

---

## Execution Order

1. **Setup** (one-time):
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Download** (one-time):
   ```bash
   python preprocessing/download_iemocap.py
   python preprocessing/download_mosei.py
   ```

3. **Feature Extraction** (one-time):
   ```bash
   python preprocessing/extract_features.py --dataset both
   ```

4. **Dataset Creation** (one-time):
   ```bash
   python preprocessing/create_dataset.py --dataset both
   ```

5. **Training** (one-time per model):
   ```bash
   python training/train_incongruence.py
   python training/train_temporal.py
   ```

6. **Evaluation** (repeatable):
   ```bash
   python evaluate.py --dataset both
   ```

7. **Analysis** (repeatable):
   ```bash
   jupyter notebook analysis/visualization.ipynb
   ```

Or use the automated script:
```bash
./run_pipeline.sh
```

---

## Configuration Files

All configuration is centralized in `utils/config.py`:

- **Paths**: Data directories, checkpoints, results
- **Dimensions**: Feature dimensions for each modality
- **Hyperparameters**: Learning rates, batch sizes, epochs
- **Model configs**: Architecture specifications
- **Device**: CPU/GPU settings

To modify settings, edit `utils/config.py` before running scripts.

---

## Output Files

### After Feature Extraction:
- `data/processed/iemocap_features.pkl`
- `data/processed/mosei_features.pkl`

### After Dataset Creation:
- `data/processed/iemocap_train.pkl`
- `data/processed/iemocap_val.pkl`
- `data/processed/iemocap_temporal.pkl`
- `data/processed/mosei_test.pkl`

### After Training:
- `checkpoints/incongruence_encoder_best.pth`
- `checkpoints/temporal_model_best.pth`
- `logs/incongruence_encoder/` (TensorBoard logs)
- `logs/temporal_model/` (TensorBoard logs)

### After Evaluation:
- `results/eci_scores_iemocap.csv`
- `results/eci_scores_mosei.csv`
- `results/cross_dataset_comparison.json`
- `results/summary_statistics.csv`
- `results/*.png` (visualization plots)

---

## Notes

- All Python scripts include docstrings and type hints
- Error handling is implemented throughout
- Progress bars (tqdm) for long-running operations
- Checkpointing for resumable training
- Logging for debugging and monitoring
- Modular design for easy extension

---

## Support

For issues with specific files:
1. Check file docstrings for usage examples
2. Run with `--help` flag for CLI scripts
3. Review `IMPLEMENTATION_GUIDE.md` for troubleshooting