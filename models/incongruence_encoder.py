"""
Incongruence Encoder - Cross-Modal Learning
Learns to detect disagreement between face, audio, and text modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import INCONGRUENCE_CONFIG, FACE_DIM, AUDIO_DIM, TEXT_DIM, DEVICE


class IncongruenceEncoder(nn.Module):
    """
    Neural network that learns cross-modal incongruence
    Trained via self-supervised contrastive learning
    
    Architecture:
        Input: Concatenated [face, audio, text] features
        Hidden: Multi-layer MLP with dropout
        Output: Incongruence representation vector
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize incongruence encoder
        
        Args:
            config: Configuration dict (uses INCONGRUENCE_CONFIG if None)
        """
        super(IncongruenceEncoder, self).__init__()
        
        self.config = config or INCONGRUENCE_CONFIG
        
        # Input dimension: sum of all modalities
        self.input_dim = self.config['input_dim']
        self.hidden_dims = self.config['hidden_dims']
        self.output_dim = self.config['output_dim']
        self.dropout = self.config['dropout']
        
        # Build MLP layers
        layers = []
        
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation for embedding space)
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # L2 normalization for embeddings
        self.normalize = True
    
    def forward(self, face: torch.Tensor, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            face: Face features (B, face_dim)
            audio: Audio features (B, audio_dim)
            text: Text features (B, text_dim)
        
        Returns:
            Incongruence embedding (B, output_dim)
        """
        # Concatenate all modalities
        multimodal = torch.cat([face, audio, text], dim=1)
        
        # Pass through encoder
        embedding = self.encoder(multimodal)
        
        # L2 normalize for contrastive learning
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def compute_incongruence_score(self, face: torch.Tensor, audio: torch.Tensor, 
                                   text: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar incongruence score
        Higher score = more incongruence
        
        This is computed as the variance in pairwise similarities
        """
        with torch.no_grad():
            # Get embeddings for each modality alone (zero out others)
            batch_size = face.size(0)
            zeros_face = torch.zeros_like(face)
            zeros_audio = torch.zeros_like(audio)
            zeros_text = torch.zeros_like(text)
            
            # Individual modality embeddings
            face_emb = self(face, zeros_audio, zeros_text)
            audio_emb = self(zeros_face, audio, zeros_text)
            text_emb = self(zeros_face, zeros_audio, text)
            
            # Compute pairwise cosine similarities
            sim_fa = F.cosine_similarity(face_emb, audio_emb, dim=1)
            sim_ft = F.cosine_similarity(face_emb, text_emb, dim=1)
            sim_at = F.cosine_similarity(audio_emb, text_emb, dim=1)
            
            # Incongruence = variance in similarities
            # Stack similarities
            sims = torch.stack([sim_fa, sim_ft, sim_at], dim=1)  # (B, 3)
            
            # Compute variance (high variance = high incongruence)
            incongruence = torch.var(sims, dim=1)
        
        return incongruence


class ContrastiveDataset(torch.utils.data.Dataset):
    """
    Dataset for self-supervised contrastive learning
    Creates positive (aligned) and negative (misaligned) pairs
    """
    
    def __init__(self, face_features: np.ndarray, audio_features: np.ndarray, 
                 text_features: np.ndarray):
        """
        Args:
            face_features: (N, face_dim)
            audio_features: (N, audio_dim)
            text_features: (N, text_dim)
        """
        self.face = torch.FloatTensor(face_features)
        self.audio = torch.FloatTensor(audio_features)
        self.text = torch.FloatTensor(text_features)
        
        self.n_samples = len(face_features)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Return aligned sample and create negatives via shuffling
        
        Returns:
            (face, audio, text,           # Positive (aligned)
             face_neg, audio_neg, text_neg)  # Negative (one modality shuffled)
        """
        # Positive: aligned sample
        face_pos = self.face[idx]
        audio_pos = self.audio[idx]
        text_pos = self.text[idx]
        
        # Negative: shuffle one modality randomly
        # Randomly choose which modality to shuffle
        shuffle_choice = np.random.randint(0, 3)
        
        # Get random different index
        neg_idx = idx
        while neg_idx == idx:
            neg_idx = np.random.randint(0, self.n_samples)
        
        # Create negative by replacing one modality
        if shuffle_choice == 0:  # Shuffle face
            face_neg = self.face[neg_idx]
            audio_neg = audio_pos
            text_neg = text_pos
        elif shuffle_choice == 1:  # Shuffle audio
            face_neg = face_pos
            audio_neg = self.audio[neg_idx]
            text_neg = text_pos
        else:  # Shuffle text
            face_neg = face_pos
            audio_neg = audio_pos
            text_neg = self.text[neg_idx]
        
        return face_pos, audio_pos, text_pos, face_neg, audio_neg, text_neg


def create_contrastive_loss(anchor_emb: torch.Tensor, positive_emb: torch.Tensor, 
                           negative_emb: torch.Tensor, margin: float = 1.0, 
                           temperature: float = 0.5) -> torch.Tensor:
    """
    Triplet contrastive loss
    
    Goal: 
        - Maximize similarity between aligned samples (anchor, positive)
        - Minimize similarity between misaligned samples (anchor, negative)
    
    Args:
        anchor_emb: Original multimodal embedding (B, D)
        positive_emb: Aligned multimodal embedding (B, D)
        negative_emb: Misaligned multimodal embedding (B, D)
        margin: Triplet margin
        temperature: Temperature for similarity scaling
    
    Returns:
        Contrastive loss
    """
    # Compute cosine similarities
    pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=1)
    neg_sim = F.cosine_similarity(anchor_emb, negative_emb, dim=1)
    
    # Scale by temperature
    pos_sim = pos_sim / temperature
    neg_sim = neg_sim / temperature
    
    # Triplet loss: max(0, margin + neg_sim - pos_sim)
    loss = torch.clamp(margin + neg_sim - pos_sim, min=0.0)
    
    return loss.mean()


class IncongruenceTrainer:
    """
    Trainer for incongruence encoder using contrastive learning
    """
    
    def __init__(self, model: IncongruenceEncoder, device: str = DEVICE):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader, 
                   optimizer: torch.optim.Optimizer, 
                   margin: float = 1.0, temperature: float = 0.5) -> float:
        """
        Train for one epoch
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            face_pos, audio_pos, text_pos, face_neg, audio_neg, text_neg = batch
            
            # Move to device
            face_pos = face_pos.to(self.device)
            audio_pos = audio_pos.to(self.device)
            text_pos = text_pos.to(self.device)
            face_neg = face_neg.to(self.device)
            audio_neg = audio_neg.to(self.device)
            text_neg = text_neg.to(self.device)
            
            # Forward pass
            # Anchor and positive are the same (both aligned)
            anchor_emb = self.model(face_pos, audio_pos, text_pos)
            positive_emb = self.model(face_pos, audio_pos, text_pos)
            negative_emb = self.model(face_neg, audio_neg, text_neg)
            
            # Compute loss
            loss = create_contrastive_loss(
                anchor_emb, positive_emb, negative_emb,
                margin=margin, temperature=temperature
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: torch.utils.data.DataLoader,
                margin: float = 1.0, temperature: float = 0.5) -> float:
        """
        Validate the model
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                face_pos, audio_pos, text_pos, face_neg, audio_neg, text_neg = batch
                
                # Move to device
                face_pos = face_pos.to(self.device)
                audio_pos = audio_pos.to(self.device)
                text_pos = text_pos.to(self.device)
                face_neg = face_neg.to(self.device)
                audio_neg = audio_neg.to(self.device)
                text_neg = text_neg.to(self.device)
                
                # Forward pass
                anchor_emb = self.model(face_pos, audio_pos, text_pos)
                positive_emb = self.model(face_pos, audio_pos, text_pos)
                negative_emb = self.model(face_neg, audio_neg, text_neg)
                
                # Compute loss
                loss = create_contrastive_loss(
                    anchor_emb, positive_emb, negative_emb,
                    margin=margin, temperature=temperature
                )
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)


# ==================== TESTING ====================

def test_incongruence_encoder():
    """Test incongruence encoder"""
    print("Testing Incongruence Encoder...")
    
    # Create dummy data
    batch_size = 16
    face = torch.randn(batch_size, FACE_DIM)
    audio = torch.randn(batch_size, AUDIO_DIM)
    text = torch.randn(batch_size, TEXT_DIM)
    
    # Create model
    model = IncongruenceEncoder()
    
    # Forward pass
    embedding = model(face, audio, text)
    print(f"✓ Output embedding shape: {embedding.shape}")
    print(f"✓ Expected shape: ({batch_size}, {INCONGRUENCE_CONFIG['output_dim']})")
    
    # Test incongruence score
    scores = model.compute_incongruence_score(face, audio, text)
    print(f"✓ Incongruence scores shape: {scores.shape}")
    print(f"✓ Sample scores: {scores[:5]}")
    
    print("\n✓ Incongruence encoder test passed!")


if __name__ == "__main__":
    test_incongruence_encoder()