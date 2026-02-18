"""
Utility functions for metrics, evaluation, and analysis
"""

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mutual_info_score
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== CROSS-MODAL INCONGRUENCE METRICS ====================

def compute_cross_modal_distance(emb1: np.ndarray, emb2: np.ndarray, metric: str = 'cosine') -> float:
    """
    Compute distance between two modality embeddings
    
    Args:
        emb1: First embedding (N, D)
        emb2: Second embedding (N, D)
        metric: 'cosine', 'euclidean', or 'correlation'
    
    Returns:
        Distance score
    """
    if metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)
        similarity = np.sum((emb1 / norm1) * (emb2 / norm2), axis=1)
        return np.mean(1 - similarity)
    
    elif metric == 'euclidean':
        return np.mean(np.linalg.norm(emb1 - emb2, axis=1))
    
    elif metric == 'correlation':
        correlations = [stats.pearsonr(emb1[i], emb2[i])[0] for i in range(len(emb1))]
        return np.mean(1 - np.abs(correlations))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_multimodal_incongruence(face: np.ndarray, audio: np.ndarray, text: np.ndarray) -> Dict[str, float]:
    """
    Compute pairwise incongruence between all modality pairs
    
    Returns:
        Dictionary with face-audio, face-text, audio-text distances
    """
    return {
        'face_audio': compute_cross_modal_distance(face, audio),
        'face_text': compute_cross_modal_distance(face, text),
        'audio_text': compute_cross_modal_distance(audio, text),
        'mean': np.mean([
            compute_cross_modal_distance(face, audio),
            compute_cross_modal_distance(face, text),
            compute_cross_modal_distance(audio, text)
        ])
    }


# ==================== TEMPORAL STABILITY METRICS ====================

def compute_temporal_variance(sequence: np.ndarray, window_size: int = 5) -> float:
    """
    Compute temporal variance using sliding window
    Lower variance = more regulation
    
    Args:
        sequence: (T, D) temporal sequence
        window_size: Window for variance computation
    
    Returns:
        Mean temporal variance
    """
    if len(sequence) < window_size:
        return np.var(sequence)
    
    variances = []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        variances.append(np.var(window))
    
    return np.mean(variances)


def compute_dampening_rate(sequence: np.ndarray) -> float:
    """
    Compute how quickly signal dampens over time
    Higher rate = stronger regulation
    
    Uses exponential decay fitting
    """
    if len(sequence) < 3:
        return 0.0
    
    t = np.arange(len(sequence))
    
    # Fit exponential decay: y = a * exp(-b * t)
    # Use log transform: log(y) = log(a) - b * t
    try:
        # Avoid log(0) by adding small epsilon
        log_seq = np.log(np.abs(sequence) + 1e-8)
        slope, _, _, _, _ = stats.linregress(t, log_seq)
        return -slope  # Negative slope = decay
    except:
        return 0.0


# ==================== CONTRASTIVE LEARNING METRICS ====================

def contrastive_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, 
                     margin: float = 1.0, temperature: float = 0.5) -> torch.Tensor:
    """
    Triplet contrastive loss for self-supervised learning
    
    Args:
        anchor: Original multimodal embedding (B, D)
        positive: Aligned multimodal embedding (B, D)
        negative: Misaligned multimodal embedding (B, D)
        margin: Margin for triplet loss
        temperature: Temperature for similarity scaling
    
    Returns:
        Contrastive loss
    """
    # Compute similarities
    pos_sim = torch.nn.functional.cosine_similarity(anchor, positive, dim=1)
    neg_sim = torch.nn.functional.cosine_similarity(anchor, negative, dim=1)
    
    # Scale by temperature
    pos_sim = pos_sim / temperature
    neg_sim = neg_sim / temperature
    
    # Triplet loss: max(0, margin + neg_sim - pos_sim)
    loss = torch.clamp(margin + neg_sim - pos_sim, min=0.0)
    
    return loss.mean()


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor,
                  temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE loss (used in SimCLR, MoCo)
    
    Args:
        anchor: (B, D)
        positive: (B, D) 
        negatives: (B, N, D) where N is number of negatives
        temperature: Temperature scaling
    
    Returns:
        InfoNCE loss
    """
    # Normalize embeddings
    anchor = torch.nn.functional.normalize(anchor, dim=1)
    positive = torch.nn.functional.normalize(positive, dim=1)
    negatives = torch.nn.functional.normalize(negatives, dim=2)
    
    # Positive similarity
    pos_sim = torch.sum(anchor * positive, dim=1) / temperature  # (B,)
    
    # Negative similarities
    neg_sim = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2) / temperature  # (B, N)
    
    # Concatenate positive and negatives
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+N)
    
    # Labels: positive is always first (index 0)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    # Cross entropy loss
    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    return loss


# ==================== EVALUATION METRICS ====================

def compute_eci_statistics(eci_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for ECI scores
    
    Returns:
        Dictionary with mean, std, median, IQR, skewness, kurtosis
    """
    return {
        'mean': np.mean(eci_scores),
        'std': np.std(eci_scores),
        'median': np.median(eci_scores),
        'q25': np.percentile(eci_scores, 25),
        'q75': np.percentile(eci_scores, 75),
        'iqr': stats.iqr(eci_scores),
        'skewness': stats.skew(eci_scores),
        'kurtosis': stats.kurtosis(eci_scores),
        'min': np.min(eci_scores),
        'max': np.max(eci_scores)
    }


def compare_distributions(scores1: np.ndarray, scores2: np.ndarray, 
                         name1: str = 'Distribution 1', name2: str = 'Distribution 2') -> Dict:
    """
    Compare two ECI distributions (e.g., IEMOCAP vs MOSEI)
    
    Returns:
        Statistical test results
    """
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(scores1, scores2)
    
    # Mann-Whitney U test
    u_stat, u_pval = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
    
    # Effect size (Cohen's d)
    cohens_d = (np.mean(scores1) - np.mean(scores2)) / np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
    
    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'mannwhitney_u': u_stat,
        'mannwhitney_pvalue': u_pval,
        'cohens_d': cohens_d,
        f'{name1}_mean': np.mean(scores1),
        f'{name2}_mean': np.mean(scores2),
        f'{name1}_std': np.std(scores1),
        f'{name2}_std': np.std(scores2)
    }


# ==================== VISUALIZATION HELPERS ====================

def plot_eci_distribution(scores: np.ndarray, title: str = 'ECI Distribution', 
                         save_path: str = None):
    """
    Plot ECI score distribution
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
    axes[0].axvline(np.median(scores), color='green', linestyle='--', label=f'Median: {np.median(scores):.3f}')
    axes[0].set_xlabel('Expressive Control Index (ECI)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{title} - Histogram')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot(scores, vert=True)
    axes[1].set_ylabel('Expressive Control Index (ECI)')
    axes[1].set_title(f'{title} - Box Plot')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_temporal_sequence(sequence: np.ndarray, title: str = 'Temporal Dynamics',
                          save_path: str = None):
    """
    Plot temporal sequence of incongruence or ECI
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    t = np.arange(len(sequence))
    ax.plot(t, sequence, marker='o', linewidth=2, markersize=4)
    ax.fill_between(t, sequence, alpha=0.3)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# ==================== LOGGING UTILITIES ====================

def log_metrics(metrics: Dict, step: int, writer=None):
    """
    Log metrics to console and TensorBoard
    
    Args:
        metrics: Dictionary of metric_name -> value
        step: Training step or epoch
        writer: TensorBoard writer (optional)
    """
    # Console logging
    metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    print(f"Step {step} | {metric_str}")
    
    # TensorBoard logging
    if writer is not None:
        for name, value in metrics.items():
            writer.add_scalar(name, value, step)


def save_results(results: Dict, filename: str):
    """
    Save results to JSON file
    """
    import json
    from pathlib import Path
    
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filepath}")