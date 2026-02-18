"""
Global configuration for emotion regulation analysis project
"""

import os
from pathlib import Path

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IEMOCAP_DIR = DATA_DIR / "iemocap"
MOSEI_DIR = DATA_DIR / "mosei"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, IEMOCAP_DIR, MOSEI_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== FEATURE DIMENSIONS ====================
FACE_DIM = 468 * 2  # MediaPipe landmarks (x, y coordinates)
AUDIO_DIM = 40  # MFCC (13) + pitch (1) + energy (1) + deltas (25)
TEXT_DIM = 768  # RoBERTa embedding dimension

# ==================== MODEL HYPERPARAMETERS ====================

# Incongruence Encoder
INCONGRUENCE_CONFIG = {
    'input_dim': FACE_DIM + AUDIO_DIM + TEXT_DIM,
    'hidden_dims': [512, 256, 128],
    'output_dim': 64,
    'dropout': 0.3,
    'activation': 'relu'
}

# Temporal Model (Bi-LSTM)
TEMPORAL_CONFIG = {
    'input_dim': 64,  # Output from incongruence encoder
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': True,
    'output_dim': 1  # ECI score
}

# ==================== TRAINING HYPERPARAMETERS ====================

# Incongruence Encoder Training
INCONGRUENCE_TRAIN = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 15,
    'optimizer': 'adam',
    'weight_decay': 1e-5,
    'scheduler': 'cosine',
    'warmup_epochs': 2
}

# Temporal Model Training
TEMPORAL_TRAIN = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 20,
    'optimizer': 'adam',
    'weight_decay': 1e-5,
    'scheduler': 'cosine',
    'warmup_epochs': 3,
    'sequence_length': 20  # Temporal window size
}

# ==================== DATA PROCESSING ====================

# Audio Processing
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'n_mfcc': 13,
    'n_fft': 2048,
    'hop_length': 512,
    'window_length': 512
}

# Face Processing
FACE_CONFIG = {
    'max_faces': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Text Processing
TEXT_CONFIG = {
    'model_name': 'SamLowe/roberta-base-go_emotions',
    'max_length': 128,
    'use_emotion_head': True
}

# ==================== DATASET SETTINGS ====================

# Train/Val Split
TRAIN_VAL_SPLIT = 0.85

# Augmentation (for robustness)
AUGMENTATION = {
    'temporal_shift': True,
    'noise_injection': False,  # Conservative for regulation analysis
    'modality_dropout': 0.1
}

# ==================== EVALUATION SETTINGS ====================

# Ablation Study
ABLATION_CONFIG = {
    'modalities': ['face', 'audio', 'text'],
    'temporal': [True, False],
    'sequence_lengths': [10, 20, 30]
}

# Visualization
VIS_CONFIG = {
    'plot_style': 'seaborn-v0_8-darkgrid',
    'figure_size': (12, 6),
    'dpi': 150,
    'save_format': 'png'
}

# ==================== SYSTEM SETTINGS ====================

# Device Configuration
DEVICE = 'cpu'  # M4 Mac - CPU optimized
NUM_WORKERS = 4  # For DataLoader
PIN_MEMORY = False

# Reproducibility
RANDOM_SEED = 42

# Logging
LOG_INTERVAL = 10  # Log every N batches
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs

# ==================== DATASET URLs ====================

# IEMOCAP (Kaggle)
IEMOCAP_KAGGLE = "jamaliasultanajisha/iemocap-full"

# CMU-MOSEI
MOSEI_URL = "http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip"

# ==================== EMOTION LABELS (for analysis only) ====================

IEMOCAP_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'frustrated', 'excited']

# ==================== HELPER FUNCTIONS ====================

def get_checkpoint_path(model_name: str, epoch: int = None) -> Path:
    """Get checkpoint path for a model"""
    if epoch is not None:
        return CHECKPOINT_DIR / f"{model_name}_epoch_{epoch}.pth"
    return CHECKPOINT_DIR / f"{model_name}_best.pth"

def get_results_path(filename: str) -> Path:
    """Get results file path"""
    return RESULTS_DIR / filename

def get_log_path(experiment_name: str) -> Path:
    """Get TensorBoard log path"""
    return LOGS_DIR / experiment_name