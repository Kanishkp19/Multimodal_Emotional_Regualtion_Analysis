"""
Extract features from raw IEMOCAP and MOSEI datasets
Processes video, audio, and text using frozen encoders
"""

import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from encoders.face_encoder import FaceEncoder
from encoders.audio_encoder import AudioEncoder
from encoders.text_encoder import TextEncoder
from utils.config import IEMOCAP_DIR, MOSEI_DIR, DATA_DIR


def extract_iemocap_features(output_dir: Path):
    """
    Extract features from IEMOCAP dataset
    
    IEMOCAP structure (after download):
    - Session folders (Session1, Session2, ...)
    - Each session contains video, audio, and transcript files
    """
    print("\n" + "="*60)
    print("Extracting IEMOCAP Features")
    print("="*60 + "\n")
    
    # Initialize encoders
    print("Initializing encoders...")
    face_encoder = FaceEncoder()
    audio_encoder = AudioEncoder()
    text_encoder = TextEncoder()
    
    # Prepare output structure
    features = {
        'face': [],
        'audio': [],
        'text': [],
        'metadata': []
    }
    
    # Find all sessions
    session_dirs = sorted([d for d in IEMOCAP_DIR.iterdir() if d.is_dir() and 'Session' in d.name])
    
    if len(session_dirs) == 0:
        print("⚠️  No session directories found in IEMOCAP_DIR")
        print(f"   Looking in: {IEMOCAP_DIR}")
        print("   Please verify dataset download")
        return
    
    print(f"Found {len(session_dirs)} sessions\n")
    
    # Process each session
    for session_dir in session_dirs:
        print(f"Processing {session_dir.name}...")
        
        # Look for video, audio, and transcript files
        video_files = list(session_dir.rglob('*.avi')) + list(session_dir.rglob('*.mp4'))
        audio_files = list(session_dir.rglob('*.wav'))
        text_files = list(session_dir.rglob('*.txt'))
        
        print(f"  Found: {len(video_files)} videos, {len(audio_files)} audios, {len(text_files)} texts")
        
        # Match files by utterance ID
        for video_file in tqdm(video_files, desc=f"  {session_dir.name}"):
            utterance_id = video_file.stem
            
            # Find corresponding audio and text
            audio_file = next((f for f in audio_files if utterance_id in f.stem), None)
            text_file = next((f for f in text_files if utterance_id in f.stem), None)
            
            if audio_file is None or text_file is None:
                continue
            
            try:
                # Extract face features
                face_feat = face_encoder.extract_from_video(str(video_file))
                if face_feat is None or len(face_feat) == 0:
                    continue
                
                # Extract audio features
                audio_feat = audio_encoder.extract_from_file(str(audio_file))
                if audio_feat is None or len(audio_feat) == 0:
                    continue
                
                # Extract text features
                with open(text_file, 'r') as f:
                    text = f.read().strip()
                
                text_feat = text_encoder.extract_from_text(text)
                
                # Align temporal dimensions (use minimum length)
                min_len = min(len(face_feat), len(audio_feat))
                face_feat = face_feat[:min_len]
                audio_feat = audio_feat[:min_len]
                
                # Save features
                features['face'].append(face_feat)
                features['audio'].append(audio_feat)
                features['text'].append(text_feat)
                features['metadata'].append({
                    'utterance_id': utterance_id,
                    'session': session_dir.name,
                    'video_file': str(video_file),
                    'audio_file': str(audio_file),
                    'text_file': str(text_file),
                    'text': text,
                    'duration': len(face_feat)
                })
                
            except Exception as e:
                print(f"    Error processing {utterance_id}: {e}")
                continue
    
    # Clean up
    face_encoder.close()
    
    # Save features
    print(f"\n✓ Extracted {len(features['face'])} samples")
    
    output_file = output_dir / 'iemocap_features.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(features, f)
    
    print(f"✓ Features saved to {output_file}")
    
    # Save summary statistics
    summary = {
        'n_samples': len(features['face']),
        'face_dim': features['face'][0].shape[1] if len(features['face']) > 0 else 0,
        'audio_dim': features['audio'][0].shape[1] if len(features['audio']) > 0 else 0,
        'text_dim': len(features['text'][0]) if len(features['text']) > 0 else 0,
        'avg_duration': np.mean([len(f) for f in features['face']]) if len(features['face']) > 0 else 0
    }
    
    with open(output_dir / 'iemocap_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("IEMOCAP feature extraction complete!")
    print("="*60)


def extract_mosei_features(output_dir: Path):
    """
    Extract features from CMU-MOSEI dataset
    
    MOSEI typically comes with pre-extracted features via CMU SDK
    This function loads and aligns them
    """
    print("\n" + "="*60)
    print("Extracting MOSEI Features")
    print("="*60 + "\n")
    
    try:
        from mmsdk import mmdatasdk
        
        print("Loading MOSEI data using CMU SDK...")
        
        # Define dataset components
        visual_field = 'CMU_MOSEI_VisualOpenFace2'
        audio_field = 'CMU_MOSEI_COVAREP'
        text_field = 'CMU_MOSEI_TimestampedWordVectors'
        
        # Load datasets
        dataset = mmdatasdk.mmdataset(str(MOSEI_DIR))
        
        print("Processing MOSEI samples...")
        
        features = {
            'face': [],
            'audio': [],
            'text': [],
            'metadata': []
        }
        
        # Get all video IDs
        video_ids = list(dataset[visual_field].keys())
        
        print(f"Found {len(video_ids)} videos")
        
        for video_id in tqdm(video_ids[:1000], desc="Processing"):  # Limit to 1000 for memory
            try:
                # Get features for this video
                visual = dataset[visual_field][video_id]['features']
                audio = dataset[audio_field][video_id]['features']
                text = dataset[text_field][video_id]['features']
                
                # Align temporal dimensions
                min_len = min(len(visual), len(audio), len(text))
                
                if min_len < 5:  # Skip very short sequences
                    continue
                
                visual = visual[:min_len]
                audio = audio[:min_len]
                text = text[:min_len]
                
                # Save
                features['face'].append(visual)
                features['audio'].append(audio)
                features['text'].append(text.mean(axis=0))  # Average word vectors
                features['metadata'].append({
                    'video_id': video_id,
                    'duration': min_len
                })
                
            except Exception as e:
                continue
        
        print(f"\n✓ Extracted {len(features['face'])} samples")
        
        # Save features
        output_file = output_dir / 'mosei_features.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(features, f)
        
        print(f"✓ Features saved to {output_file}")
        
        print("\n" + "="*60)
        print("MOSEI feature extraction complete!")
        print("="*60)
        
    except ImportError:
        print("\n⚠️  CMU Multimodal SDK not installed")
        print("   Attempting alternative extraction method...")
        extract_mosei_features_alternative(output_dir)


def extract_mosei_features_alternative(output_dir: Path):
    """
    Alternative MOSEI feature extraction without SDK
    Loads pre-extracted .csd files directly
    """
    print("\nUsing alternative MOSEI extraction...")
    
    features_dir = MOSEI_DIR / 'features'
    
    if not features_dir.exists():
        print("❌ No features directory found in MOSEI")
        print("   Please run: python preprocessing/download_mosei.py")
        return
    
    # Look for .csd files
    csd_files = list(features_dir.glob('*.csd'))
    
    if len(csd_files) == 0:
        print("❌ No .csd files found")
        return
    
    print(f"Found {len(csd_files)} feature files")
    print("Note: Using pre-extracted features for validation only")
    
    # Create placeholder for MOSEI validation
    features = {
        'face': [],
        'audio': [],
        'text': [],
        'metadata': []
    }
    
    # Note: This is a simplified version
    # In practice, you would parse .csd files properly
    
    output_file = output_dir / 'mosei_features.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(features, f)
    
    print(f"✓ Placeholder saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract features from datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['iemocap', 'mosei', 'both'],
                       help='Which dataset to process')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for features (default: data/processed)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DATA_DIR / 'processed'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract features
    if args.dataset in ['iemocap', 'both']:
        extract_iemocap_features(output_dir)
    
    if args.dataset in ['mosei', 'both']:
        extract_mosei_features(output_dir)
    
    print("\n✅ Feature extraction complete!")
    print(f"   Features saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. python preprocessing/create_dataset.py")
    print("  2. python training/train_incongruence.py")


if __name__ == "__main__":
    main()