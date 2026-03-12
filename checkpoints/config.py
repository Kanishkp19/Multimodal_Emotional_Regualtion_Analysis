import torch
from pathlib import Path

# Colab local paths
LOCAL_BASE    = Path('/content/emotion_regulation')
IEMOCAP_DIR   = LOCAL_BASE / 'IEMOCAP'
MOSEI_DIR     = LOCAL_BASE / 'MOSEI'
PROCESSED_DIR = LOCAL_BASE / 'processed'

# Google Drive paths (checkpoints + results only)
DRIVE_BASE     = Path('/content/drive/MyDrive/emotion_regulation')
CHECKPOINT_DIR = DRIVE_BASE / 'checkpoints'
RESULTS_DIR    = DRIVE_BASE / 'results'
LOGS_DIR       = DRIVE_BASE / 'logs'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Only Session1 is used — 80/20 split
TRAIN_SESSION      = 'Session1'
MOSEI_EVAL_SAMPLES = 500

# Feature dimensions (verified from actual extraction)
FACE_DIM  = 944
AUDIO_DIM = 46   # actual extracted dim = 44 (not 42)
TEXT_DIM  = 796
INPUT_DIM = 1786

INCONGRUENCE_CONFIG = {
    'input_dim':   INPUT_DIM,
    'hidden_dims': [512, 256, 128],
    'output_dim':  64,
    'dropout':     0.3
}
TEMPORAL_CONFIG    = {'input_dim': 64, 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.3}
INCONGRUENCE_TRAIN = {'batch_size': 32, 'learning_rate': 1e-3, 'epochs': 15, 'weight_decay': 1e-5}
TEMPORAL_TRAIN     = {'batch_size': 32, 'learning_rate': 1e-3, 'epochs': 20, 'weight_decay': 1e-5, 'sequence_length': 20}
AUDIO_CONFIG       = {'sample_rate': 16000, 'n_mfcc': 13, 'n_fft': 2048, 'hop_length': 512}
TEXT_CONFIG        = {'model_name': 'SamLowe/roberta-base-go_emotions', 'max_length': 128}
RANDOM_SEED        = 42

print(f'Config loaded. Device: {DEVICE}  |  Input dim: {INPUT_DIM}')
