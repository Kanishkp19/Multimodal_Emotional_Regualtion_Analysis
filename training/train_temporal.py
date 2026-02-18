"""
Train Temporal Model for ECI Computation
Aggregates incongruence sequences over time
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
from pathlib import Path
import argparse
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.temporal_model import TemporalModel, TemporalDataset, TemporalTrainer
from models.incongruence_encoder import IncongruenceEncoder
from utils.config import (TEMPORAL_TRAIN, DATA_DIR, CHECKPOINT_DIR, 
                          LOGS_DIR, DEVICE, RANDOM_SEED)
from utils.metrics import log_metrics


def load_temporal_data(data_path: Path) -> tuple:
    """Load temporal (sequence) data"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['face'], data['audio'], data['text']


def extract_incongruence_features(face_seq, audio_seq, text_seq, 
                                  incongruence_model: IncongruenceEncoder) -> np.ndarray:
    """
    Extract incongruence features from raw multimodal sequences
    
    Args:
        face_seq: List of face sequences (each is (T, D_face))
        audio_seq: List of audio sequences (each is (T, D_audio))
        text_seq: List of text features (each is (D_text,))
        incongruence_model: Trained incongruence encoder
    
    Returns:
        List of incongruence sequences
    """
    print("Extracting incongruence features...")
    
    incongruence_model.eval()
    
    incongruence_sequences = []
    
    for i in tqdm(range(len(face_seq))):
        face = face_seq[i]  # (T, D_face)
        audio = audio_seq[i]  # (T, D_audio)
        text = text_seq[i]  # (D_text,)
        
        # Process each frame
        T = len(face)
        frame_features = []
        
        with torch.no_grad():
            for t in range(T):
                # Get frame features
                face_frame = torch.FloatTensor(face[t]).unsqueeze(0).to(DEVICE)
                audio_frame = torch.FloatTensor(audio[t]).unsqueeze(0).to(DEVICE)
                text_frame = torch.FloatTensor(text).unsqueeze(0).to(DEVICE)
                
                # Get incongruence embedding
                incong_feat = incongruence_model(face_frame, audio_frame, text_frame)
                frame_features.append(incong_feat.cpu().numpy()[0])
        
        incongruence_sequences.append(np.array(frame_features))
    
    return incongruence_sequences


def train(args):
    """Main training function"""
    
    print("\n" + "="*60)
    print("Training Temporal Model")
    print("="*60 + "\n")
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Load incongruence encoder
    print("Loading incongruence encoder...")
    incongruence_model = IncongruenceEncoder()
    checkpoint = torch.load(
        CHECKPOINT_DIR / 'incongruence_encoder_best.pth',
        map_location=DEVICE
    )
    incongruence_model.load_state_dict(checkpoint['model_state_dict'])
    incongruence_model.to(DEVICE)
    incongruence_model.eval()
    
    print("✓ Incongruence encoder loaded")
    
    # Load temporal data
    print("\nLoading temporal data...")
    processed_dir = DATA_DIR / 'processed'
    
    face_seq, audio_seq, text_seq = load_temporal_data(processed_dir / 'iemocap_temporal.pkl')
    
    print(f"✓ Loaded {len(face_seq)} sequences")
    
    # Extract incongruence features
    incongruence_sequences = extract_incongruence_features(
        face_seq, audio_seq, text_seq, incongruence_model
    )
    
    # Create dataset
    print("\nCreating temporal dataset...")
    
    # Combine all sequences into one array for dataset creation
    all_features = []
    for seq in incongruence_sequences:
        all_features.extend(seq)
    
    all_features = np.array(all_features)
    
    # Create dataset with sliding window
    dataset = TemporalDataset(
        all_features,
        sequence_length=args.sequence_length
    )
    
    print(f"✓ Created {len(dataset)} temporal samples")
    
    # Split train/val
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # Create model
    print("\nInitializing temporal model...")
    model = TemporalModel()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {n_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Loss function (MSE for regression)
    criterion = nn.MSELoss()
    
    # Create trainer
    trainer = TemporalTrainer(model, device=DEVICE)
    
    # TensorBoard logging
    log_dir = LOGS_DIR / 'temporal_model'
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Training loop
    print("\nStarting training...\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
        
        # Validate
        val_loss = trainer.validate(val_loader, criterion)
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        log_metrics(metrics, epoch, writer)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = CHECKPOINT_DIR / 'temporal_model_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint periodically
        if epoch % 5 == 0:
            checkpoint_path = CHECKPOINT_DIR / f'temporal_model_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
        
        print()
    
    writer.close()
    
    print("="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CHECKPOINT_DIR}")
    print("="*60)
    
    print("\nNext step:")
    print("  python evaluate.py --dataset mosei")


def main():
    parser = argparse.ArgumentParser(description='Train Temporal Model')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=TEMPORAL_TRAIN['batch_size'],
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=TEMPORAL_TRAIN['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=TEMPORAL_TRAIN['epochs'],
                       help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=TEMPORAL_TRAIN['weight_decay'],
                       help='Weight decay')
    parser.add_argument('--sequence_length', type=int, default=TEMPORAL_TRAIN['sequence_length'],
                       help='Temporal sequence length')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()