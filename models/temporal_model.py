"""
Temporal Model - Bi-LSTM for Temporal Aggregation
Aggregates cross-modal incongruence signals over time to compute ECI
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import TEMPORAL_CONFIG, DEVICE


class TemporalModel(nn.Module):
    """
    Bi-LSTM model for temporal aggregation of incongruence signals
    
    Takes sequence of incongruence vectors and outputs:
    - Expressive Control Index (ECI) score
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize temporal model
        
        Args:
            config: Configuration dict (uses TEMPORAL_CONFIG if None)
        """
        super(TemporalModel, self).__init__()
        
        self.config = config or TEMPORAL_CONFIG
        
        self.input_dim = self.config['input_dim']
        self.hidden_dim = self.config['hidden_dim']
        self.num_layers = self.config['num_layers']
        self.dropout = self.config['dropout']
        self.bidirectional = self.config['bidirectional']
        self.output_dim = self.config['output_dim']
        
        # Bi-LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        
        # Attention mechanism for temporal pooling
        self.attention = TemporalAttention(lstm_output_dim)
        
        # Output layers for ECI score
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_dim // 2, self.output_dim),
            nn.Sigmoid()  # ECI ∈ [0, 1]
        )
    
    def forward(self, incongruence_sequence: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            incongruence_sequence: (B, T, D) where T is sequence length
            lengths: (B,) actual lengths for variable-length sequences
        
        Returns:
            ECI scores (B, 1)
        """
        batch_size, seq_len, _ = incongruence_sequence.size()
        
        # Pack sequences if lengths provided (for variable-length sequences)
        if lengths is not None:
            packed_input = nn.utils.rnn.pack_padded_sequence(
                incongruence_sequence, lengths.cpu(), 
                batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            # Process full sequences
            lstm_out, (h_n, c_n) = self.lstm(incongruence_sequence)
        
        # Apply attention to get weighted temporal representation
        # Shape: (B, lstm_output_dim)
        attended = self.attention(lstm_out, lengths)
        
        # Generate ECI score
        eci = self.fc(attended)
        
        return eci
    
    def extract_temporal_features(self, incongruence_sequence: torch.Tensor) -> dict:
        """
        Extract interpretable temporal features for analysis
        
        Returns:
            Dictionary with temporal dynamics metrics
        """
        with torch.no_grad():
            # Convert to numpy
            seq = incongruence_sequence.cpu().numpy()
            
            features = {}
            
            # Temporal variance (lower = more regulated)
            features['temporal_variance'] = np.var(seq, axis=1).mean()
            
            # Temporal smoothness (frame-to-frame change)
            if seq.shape[1] > 1:
                diffs = np.diff(seq, axis=1)
                features['temporal_smoothness'] = np.linalg.norm(diffs, axis=2).mean()
            else:
                features['temporal_smoothness'] = 0.0
            
            # Temporal range (max - min)
            features['temporal_range'] = (seq.max(axis=1) - seq.min(axis=1)).mean()
            
            # Early vs late dynamics (first half vs second half variance)
            mid_point = seq.shape[1] // 2
            early_var = np.var(seq[:, :mid_point, :], axis=1).mean()
            late_var = np.var(seq[:, mid_point:, :], axis=1).mean()
            features['early_late_ratio'] = early_var / (late_var + 1e-8)
            
            return features


class TemporalAttention(nn.Module):
    """
    Attention mechanism for temporal pooling
    Learns to weight important time steps
    """
    
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Attention scoring network
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_out: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention to LSTM outputs
        
        Args:
            lstm_out: (B, T, H) LSTM outputs
            lengths: (B,) actual sequence lengths
        
        Returns:
            Attended representation (B, H)
        """
        # Compute attention scores
        # Shape: (B, T, 1)
        scores = self.attention_weights(lstm_out)
        
        # Create mask for padding
        if lengths is not None:
            batch_size, max_len, _ = lstm_out.size()
            mask = torch.arange(max_len, device=lstm_out.device).expand(batch_size, max_len)
            mask = mask < lengths.unsqueeze(1)
            mask = mask.unsqueeze(2)  # (B, T, 1)
            
            # Apply mask (set padding to -inf before softmax)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax to get attention weights
        # Shape: (B, T, 1)
        attention = torch.softmax(scores, dim=1)
        
        # Weighted sum of LSTM outputs
        # Shape: (B, H)
        attended = torch.sum(lstm_out * attention, dim=1)
        
        return attended


class TemporalDataset(torch.utils.data.Dataset):
    """
    Dataset for temporal model training
    Creates sequences of incongruence vectors
    """
    
    def __init__(self, incongruence_features: np.ndarray, 
                 sequence_length: int, labels: Optional[np.ndarray] = None):
        """
        Args:
            incongruence_features: (N, D) incongruence vectors
            sequence_length: Length of temporal sequences
            labels: Optional labels for supervised training
        """
        self.features = torch.FloatTensor(incongruence_features)
        self.sequence_length = sequence_length
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        
        # Create sequences
        self.sequences = []
        self.seq_labels = []
        
        for i in range(len(self.features) - sequence_length + 1):
            seq = self.features[i:i+sequence_length]
            self.sequences.append(seq)
            
            if self.labels is not None:
                # Use label from middle of sequence
                mid_idx = i + sequence_length // 2
                self.seq_labels.append(self.labels[mid_idx])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        
        if len(self.seq_labels) > 0:
            label = self.seq_labels[idx]
            return seq, label
        else:
            return seq


class TemporalTrainer:
    """
    Trainer for temporal model
    """
    
    def __init__(self, model: TemporalModel, device: str = DEVICE):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader, 
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module) -> float:
        """
        Train for one epoch
        
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            if len(batch) == 2:
                sequences, labels = batch
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
            else:
                sequences = batch.to(self.device)
                labels = None
            
            # Forward pass
            predictions = self.model(sequences)
            
            # Compute loss
            if labels is not None:
                loss = criterion(predictions, labels.unsqueeze(1))
            else:
                # Self-supervised: predict temporal stability
                # Higher variance = lower ECI (less control)
                variance = torch.var(sequences, dim=1).mean(dim=1, keepdim=True)
                target_eci = 1.0 - torch.sigmoid(variance)  # Inverse of variance
                loss = criterion(predictions, target_eci)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module) -> float:
        """
        Validate the model
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    sequences, labels = batch
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)
                else:
                    sequences = batch.to(self.device)
                    labels = None
                
                # Forward pass
                predictions = self.model(sequences)
                
                # Compute loss
                if labels is not None:
                    loss = criterion(predictions, labels.unsqueeze(1))
                else:
                    variance = torch.var(sequences, dim=1).mean(dim=1, keepdim=True)
                    target_eci = 1.0 - torch.sigmoid(variance)
                    loss = criterion(predictions, target_eci)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)


# ==================== TESTING ====================

def test_temporal_model():
    """Test temporal model"""
    print("Testing Temporal Model...")
    
    # Create dummy sequence data
    batch_size = 8
    seq_length = 20
    input_dim = TEMPORAL_CONFIG['input_dim']
    
    sequences = torch.randn(batch_size, seq_length, input_dim)
    
    # Create model
    model = TemporalModel()
    
    # Forward pass
    eci_scores = model(sequences)
    
    print(f"✓ Input shape: {sequences.shape}")
    print(f"✓ Output ECI scores shape: {eci_scores.shape}")
    print(f"✓ Sample ECI scores: {eci_scores[:5].squeeze()}")
    print(f"✓ ECI score range: [{eci_scores.min():.3f}, {eci_scores.max():.3f}]")
    
    # Test temporal features
    features = model.extract_temporal_features(sequences)
    print("\n✓ Temporal features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ Temporal model test passed!")


if __name__ == "__main__":
    test_temporal_model()