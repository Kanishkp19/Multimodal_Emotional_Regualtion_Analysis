"""
Train Incongruence Encoder via Self-Supervised Contrastive Learning
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

from models.incongruence_encoder import IncongruenceEncoder, ContrastiveDataset, IncongruenceTrainer
from utils.config import (INCONGRUENCE_TRAIN, DATA_DIR, CHECKPOINT_DIR, 
                          LOGS_DIR, DEVICE, RANDOM_SEED)
from utils.metrics import log_metrics


def load_data(data_path: Path) -> tuple:
    """Load processed data"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['face'], data['audio'], data['text']


def train(args):
    """Main training function"""
    
    print("\n" + "="*60)
    print("Training Incongruence Encoder")
    print("="*60 + "\n")
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Load data
    print("Loading data...")
    processed_dir = DATA_DIR / 'processed'
    
    train_face, train_audio, train_text = load_data(processed_dir / 'iemocap_train.pkl')
    val_face, val_audio, val_text = load_data(processed_dir / 'iemocap_val.pkl')
    
    print(f"✓ Train samples: {len(train_face)}")
    print(f"✓ Val samples: {len(val_face)}")
    
    # Create datasets
    train_dataset = ContrastiveDataset(train_face, train_audio, train_text)
    val_dataset = ContrastiveDataset(val_face, val_audio, val_text)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for M4 Mac
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    print("\nInitializing model...")
    model = IncongruenceEncoder()
    
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
    
    # Create trainer
    trainer = IncongruenceTrainer(model, device=DEVICE)
    
    # TensorBoard logging
    log_dir = LOGS_DIR / 'incongruence_encoder'
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Training loop
    print("\nStarting training...\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss = trainer.train_epoch(
            train_loader, optimizer,
            margin=args.margin,
            temperature=args.temperature
        )
        
        # Validate
        val_loss = trainer.validate(
            val_loader,
            margin=args.margin,
            temperature=args.temperature
        )
        
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
            checkpoint_path = CHECKPOINT_DIR / 'incongruence_encoder_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint periodically
        if epoch % 5 == 0:
            checkpoint_path = CHECKPOINT_DIR / f'incongruence_encoder_epoch_{epoch}.pth'
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
    print("  python training/train_temporal.py")


def main():
    parser = argparse.ArgumentParser(description='Train Incongruence Encoder')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=INCONGRUENCE_TRAIN['batch_size'],
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=INCONGRUENCE_TRAIN['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=INCONGRUENCE_TRAIN['epochs'],
                       help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=INCONGRUENCE_TRAIN['weight_decay'],
                       help='Weight decay')
    
    # Contrastive learning hyperparameters
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Triplet loss margin')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for contrastive learning')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()