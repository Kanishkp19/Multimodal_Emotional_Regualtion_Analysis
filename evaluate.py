"""
Evaluate trained models on IEMOCAP and MOSEI
Compute Expressive Control Index (ECI) scores
"""

import sys
import torch
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.incongruence_encoder import IncongruenceEncoder
from models.temporal_model import TemporalModel
from utils.config import DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, DEVICE
from utils.metrics import compute_eci_statistics, compare_distributions


def load_models():
    """Load trained models"""
    print("Loading trained models...")
    
    # Load incongruence encoder
    incongruence_model = IncongruenceEncoder()
    checkpoint = torch.load(
        CHECKPOINT_DIR / 'incongruence_encoder_best.pth',
        map_location=DEVICE
    )
    incongruence_model.load_state_dict(checkpoint['model_state_dict'])
    incongruence_model.to(DEVICE)
    incongruence_model.eval()
    
    # Load temporal model
    temporal_model = TemporalModel()
    checkpoint = torch.load(
        CHECKPOINT_DIR / 'temporal_model_best.pth',
        map_location=DEVICE
    )
    temporal_model.load_state_dict(checkpoint['model_state_dict'])
    temporal_model.to(DEVICE)
    temporal_model.eval()
    
    print("✓ Models loaded successfully")
    
    return incongruence_model, temporal_model


def compute_eci_for_dataset(dataset_path: Path, incongruence_model, temporal_model,
                            sequence_length: int = 20) -> tuple:
    """
    Compute ECI scores for a dataset
    
    Returns:
        (eci_scores, metadata)
    """
    print(f"\nProcessing {dataset_path.name}...")
    
    # Load data
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    face_seq = data['face']
    audio_seq = data['audio']
    text_seq = data['text']
    metadata = data.get('metadata', [])
    
    print(f"Loaded {len(face_seq)} sequences")
    
    eci_scores = []
    
    # Process each sequence
    for i in tqdm(range(len(face_seq)), desc="Computing ECI"):
        face = face_seq[i]
        audio = audio_seq[i]
        text = text_seq[i]
        
        # Extract incongruence sequence
        T = min(len(face), len(audio))
        incongruence_seq = []
        
        with torch.no_grad():
            for t in range(T):
                face_frame = torch.FloatTensor(face[t]).unsqueeze(0).to(DEVICE)
                audio_frame = torch.FloatTensor(audio[t]).unsqueeze(0).to(DEVICE)
                text_frame = torch.FloatTensor(text).unsqueeze(0).to(DEVICE)
                
                incong_feat = incongruence_model(face_frame, audio_frame, text_frame)
                incongruence_seq.append(incong_feat.cpu().numpy()[0])
        
        incongruence_seq = np.array(incongruence_seq)
        
        # Pad or truncate to sequence_length
        if len(incongruence_seq) < sequence_length:
            # Pad with zeros
            pad_len = sequence_length - len(incongruence_seq)
            padding = np.zeros((pad_len, incongruence_seq.shape[1]))
            incongruence_seq = np.vstack([incongruence_seq, padding])
        else:
            # Truncate
            incongruence_seq = incongruence_seq[:sequence_length]
        
        # Compute ECI using temporal model
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(incongruence_seq).unsqueeze(0).to(DEVICE)
            eci = temporal_model(seq_tensor)
            eci_scores.append(eci.item())
    
    eci_scores = np.array(eci_scores)
    
    print(f"✓ Computed {len(eci_scores)} ECI scores")
    
    return eci_scores, metadata


def evaluate(args):
    """Main evaluation function"""
    
    print("\n" + "="*60)
    print("Evaluating Emotion Regulation Analysis")
    print("="*60 + "\n")
    
    # Load models
    incongruence_model, temporal_model = load_models()
    
    # Prepare results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Evaluate on IEMOCAP (validation set)
    if args.dataset in ['iemocap', 'both']:
        iemocap_path = DATA_DIR / 'processed' / 'iemocap_temporal.pkl'
        
        if iemocap_path.exists():
            eci_scores, metadata = compute_eci_for_dataset(
                iemocap_path, incongruence_model, temporal_model,
                sequence_length=args.sequence_length
            )
            
            # Compute statistics
            stats = compute_eci_statistics(eci_scores)
            
            print("\n" + "="*60)
            print("IEMOCAP Results")
            print("="*60)
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")
            
            # Save results
            results_df = pd.DataFrame({
                'eci_score': eci_scores,
                'utterance_id': [m.get('utterance_id', f'sample_{i}') for i, m in enumerate(metadata)]
            })
            
            results_df.to_csv(RESULTS_DIR / 'eci_scores_iemocap.csv', index=False)
            print(f"\n✓ Results saved to {RESULTS_DIR / 'eci_scores_iemocap.csv'}")
            
            results['iemocap'] = {
                'scores': eci_scores,
                'statistics': stats
            }
    
    # Evaluate on MOSEI (test set)
    if args.dataset in ['mosei', 'both']:
        mosei_path = DATA_DIR / 'processed' / 'mosei_test.pkl'
        
        if mosei_path.exists():
            eci_scores, metadata = compute_eci_for_dataset(
                mosei_path, incongruence_model, temporal_model,
                sequence_length=args.sequence_length
            )
            
            # Compute statistics
            stats = compute_eci_statistics(eci_scores)
            
            print("\n" + "="*60)
            print("MOSEI Results")
            print("="*60)
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")
            
            # Save results
            results_df = pd.DataFrame({
                'eci_score': eci_scores,
                'video_id': [m.get('video_id', f'sample_{i}') for i, m in enumerate(metadata)]
            })
            
            results_df.to_csv(RESULTS_DIR / 'eci_scores_mosei.csv', index=False)
            print(f"\n✓ Results saved to {RESULTS_DIR / 'eci_scores_mosei.csv'}")
            
            results['mosei'] = {
                'scores': eci_scores,
                'statistics': stats
            }
    
    # Compare distributions if both datasets evaluated
    if 'iemocap' in results and 'mosei' in results:
        print("\n" + "="*60)
        print("Cross-Dataset Comparison")
        print("="*60)
        
        comparison = compare_distributions(
            results['iemocap']['scores'],
            results['mosei']['scores'],
            name1='IEMOCAP',
            name2='MOSEI'
        )
        
        for key, value in comparison.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Save comparison
        import json
        with open(RESULTS_DIR / 'cross_dataset_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2, default=float)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nNext steps:")
    print("  1. jupyter notebook analysis/visualization.ipynb")
    print("  2. jupyter notebook analysis/ablation_study.ipynb")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    
    parser.add_argument('--dataset', type=str, default='both',
                       choices=['iemocap', 'mosei', 'both'],
                       help='Which dataset to evaluate')
    parser.add_argument('--sequence_length', type=int, default=20,
                       help='Sequence length for temporal model')
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == "__main__":
    main()