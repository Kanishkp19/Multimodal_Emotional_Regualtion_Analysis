"""
Create processed datasets for training
Aligns features and creates train/val splits
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import DATA_DIR, TRAIN_VAL_SPLIT, RANDOM_SEED


def load_features(feature_file: Path) -> dict:
    """Load extracted features"""
    with open(feature_file, 'rb') as f:
        features = pickle.load(f)
    return features


def align_and_aggregate_features(features: dict) -> tuple:
    """
    Align temporal dimensions and aggregate to utterance level
    
    Returns:
        (face_features, audio_features, text_features)
        Each is (N, D) where N is number of utterances
    """
    print("Aligning and aggregating features...")
    
    face_list = []
    audio_list = []
    text_list = []
    
    n_samples = len(features['face'])
    
    for i in range(n_samples):
        # Get features for this sample
        face = features['face'][i]  # (T, D_face)
        audio = features['audio'][i]  # (T, D_audio)
        text = features['text'][i]  # (D_text,) - already aggregated
        
        # Aggregate temporal dimension (mean pooling)
        face_agg = np.mean(face, axis=0)
        audio_agg = np.mean(audio, axis=0)
        
        face_list.append(face_agg)
        audio_list.append(audio_agg)
        text_list.append(text)
    
    face_features = np.array(face_list)
    audio_features = np.array(audio_list)
    text_features = np.array(text_list)
    
    print(f"✓ Aggregated features:")
    print(f"  Face: {face_features.shape}")
    print(f"  Audio: {audio_features.shape}")
    print(f"  Text: {text_features.shape}")
    
    return face_features, audio_features, text_features


def create_dataset(dataset_name: str):
    """
    Create processed dataset
    
    Args:
        dataset_name: 'iemocap' or 'mosei'
    """
    print(f"\n{'='*60}")
    print(f"Creating {dataset_name.upper()} Dataset")
    print(f"{'='*60}\n")
    
    # Load extracted features
    processed_dir = DATA_DIR / 'processed'
    feature_file = processed_dir / f'{dataset_name}_features.pkl'
    
    if not feature_file.exists():
        print(f"❌ Feature file not found: {feature_file}")
        print("   Please run: python preprocessing/extract_features.py")
        return
    
    features = load_features(feature_file)
    
    if len(features['face']) == 0:
        print(f"❌ No features found in {feature_file}")
        return
    
    print(f"Loaded {len(features['face'])} samples")
    
    # Align and aggregate features
    face_agg, audio_agg, text_agg = align_and_aggregate_features(features)
    
    # Create train/val split (only for IEMOCAP)
    if dataset_name == 'iemocap':
        indices = np.arange(len(face_agg))
        
        train_idx, val_idx = train_test_split(
            indices,
            train_size=TRAIN_VAL_SPLIT,
            random_state=RANDOM_SEED,
            shuffle=True
        )
        
        # Create splits
        train_data = {
            'face': face_agg[train_idx],
            'audio': audio_agg[train_idx],
            'text': text_agg[train_idx],
            'metadata': [features['metadata'][i] for i in train_idx]
        }
        
        val_data = {
            'face': face_agg[val_idx],
            'audio': audio_agg[val_idx],
            'text': text_agg[val_idx],
            'metadata': [features['metadata'][i] for i in val_idx]
        }
        
        # Save splits
        train_file = processed_dir / 'iemocap_train.pkl'
        val_file = processed_dir / 'iemocap_val.pkl'
        
        with open(train_file, 'wb') as f:
            pickle.dump(train_data, f)
        
        with open(val_file, 'wb') as f:
            pickle.dump(val_data, f)
        
        print(f"\n✓ Created train/val splits:")
        print(f"  Train: {len(train_idx)} samples -> {train_file}")
        print(f"  Val: {len(val_idx)} samples -> {val_file}")
    
    else:  # MOSEI - no training split, validation only
        mosei_data = {
            'face': face_agg,
            'audio': audio_agg,
            'text': text_agg,
            'metadata': features['metadata']
        }
        
        mosei_file = processed_dir / 'mosei_test.pkl'
        
        with open(mosei_file, 'wb') as f:
            pickle.dump(mosei_data, f)
        
        print(f"\n✓ Created test set:")
        print(f"  Test: {len(face_agg)} samples -> {mosei_file}")
    
    # Also save raw features with temporal dimension for temporal model
    if dataset_name == 'iemocap':
        temporal_data = {
            'face': features['face'],
            'audio': features['audio'],
            'text': features['text'],
            'metadata': features['metadata']
        }
        
        temporal_file = processed_dir / 'iemocap_temporal.pkl'
        with open(temporal_file, 'wb') as f:
            pickle.dump(temporal_data, f)
        
        print(f"  Temporal: {len(features['face'])} sequences -> {temporal_file}")
    
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} dataset creation complete!")
    print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create processed datasets')
    parser.add_argument('--dataset', type=str, default='both', choices=['iemocap', 'mosei', 'both'],
                       help='Which dataset to process')
    
    args = parser.parse_args()
    
    if args.dataset in ['iemocap', 'both']:
        create_dataset('iemocap')
    
    if args.dataset in ['mosei', 'both']:
        create_dataset('mosei')
    
    print("\n✅ Dataset creation complete!")
    print("\nNext step:")
    print("  python training/train_incongruence.py")


if __name__ == "__main__":
    main()