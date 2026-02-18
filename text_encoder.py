"""
Text Encoder - RoBERTa with Emotion Features
Extracts emotional intensity and sentiment from transcripts
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from typing import List, Optional, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import TEXT_DIM, TEXT_CONFIG, DEVICE


class TextEncoder:
    """
    Frozen text encoder using RoBERTa fine-tuned on GoEmotions
    Extracts emotional embeddings and intensity scores
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize RoBERTa text encoder
        
        Args:
            config: Configuration dict (uses TEXT_CONFIG if None)
        """
        self.config = config or TEXT_CONFIG
        self.device = DEVICE
        
        model_name = self.config['model_name']
        
        print(f"Loading text encoder: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        if self.config.get('use_emotion_head', True):
            # Use emotion classification head
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            # Use base model for embeddings only
            self.model = AutoModel.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()  # Freeze model
        
        # Emotion labels (GoEmotions)
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        
        print("✓ Text Encoder initialized (RoBERTa + GoEmotions)")
    
    def extract_from_text(self, text: str) -> np.ndarray:
        """
        Extract features from single text utterance
        
        Args:
            text: Text string (transcript)
        
        Returns:
            Text features (embedding + emotion logits)
        """
        if not text or len(text.strip()) == 0:
            # Return zero vector for empty text
            return np.zeros(TEXT_DIM + len(self.emotion_labels))
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.config['max_length'],
            truncation=True,
            padding='max_length'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get [CLS] token embedding from last hidden state
            # Shape: (1, hidden_dim)
            embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()[0]
            
            # Get emotion logits if available
            if hasattr(outputs, 'logits'):
                emotion_logits = outputs.logits.cpu().numpy()[0]
            else:
                emotion_logits = np.zeros(len(self.emotion_labels))
        
        # Combine embedding and emotion logits
        features = np.concatenate([embedding, emotion_logits])
        
        return features
    
    def extract_from_texts(self, texts: List[str]) -> np.ndarray:
        """
        Extract features from multiple texts (batch processing)
        
        Args:
            texts: List of text strings
        
        Returns:
            Text features array of shape (N, D)
        """
        features_list = []
        
        for text in texts:
            features = self.extract_from_text(text)
            features_list.append(features)
        
        return np.array(features_list)
    
    def get_emotion_distribution(self, text: str) -> Dict[str, float]:
        """
        Get emotion probability distribution for text
        
        Args:
            text: Text string
        
        Returns:
            Dictionary mapping emotion labels to probabilities
        """
        features = self.extract_from_text(text)
        
        # Extract emotion logits (last N elements)
        emotion_logits = features[-len(self.emotion_labels):]
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(emotion_logits - np.max(emotion_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Create distribution dict
        distribution = {
            label: float(prob) 
            for label, prob in zip(self.emotion_labels, probs)
        }
        
        return distribution
    
    def get_emotional_intensity(self, text: str) -> float:
        """
        Compute overall emotional intensity from text
        Higher intensity = stronger emotion
        
        Args:
            text: Text string
        
        Returns:
            Intensity score [0, 1]
        """
        distribution = self.get_emotion_distribution(text)
        
        # Remove neutral probability
        non_neutral_probs = [
            prob for label, prob in distribution.items() 
            if label != 'neutral'
        ]
        
        # Intensity = sum of non-neutral emotions
        intensity = sum(non_neutral_probs)
        
        return intensity
    
    def get_dominant_emotion(self, text: str) -> str:
        """
        Get the dominant emotion from text
        
        Args:
            text: Text string
        
        Returns:
            Emotion label with highest probability
        """
        distribution = self.get_emotion_distribution(text)
        return max(distribution.items(), key=lambda x: x[1])[0]
    
    def __call__(self, input_data):
        """
        Make encoder callable
        
        Args:
            input_data: Either single text (str) or list of texts (List[str])
        
        Returns:
            Text features
        """
        if isinstance(input_data, str):
            return self.extract_from_text(input_data)
        elif isinstance(input_data, list):
            return self.extract_from_texts(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")


class TextFeatureStatistics:
    """
    Compute statistics for text features
    """
    
    @staticmethod
    def compute_emotional_variance(texts: List[str], encoder: TextEncoder) -> float:
        """
        Compute variance in emotional content across texts
        Lower variance = more regulated expression
        """
        intensities = [encoder.get_emotional_intensity(text) for text in texts]
        return np.var(intensities)
    
    @staticmethod
    def compute_emotion_shifts(texts: List[str], encoder: TextEncoder) -> int:
        """
        Count number of emotion shifts in sequence
        Fewer shifts = more regulated
        """
        emotions = [encoder.get_dominant_emotion(text) for text in texts]
        
        shifts = 0
        for i in range(1, len(emotions)):
            if emotions[i] != emotions[i-1]:
                shifts += 1
        
        return shifts
    
    @staticmethod
    def compute_sentiment_consistency(texts: List[str], encoder: TextEncoder) -> float:
        """
        Compute consistency in emotional valence
        Higher = more consistent (regulated)
        """
        if len(texts) < 2:
            return 1.0
        
        # Define valence for emotions
        positive_emotions = {'admiration', 'amusement', 'approval', 'caring', 'excitement', 
                           'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'}
        negative_emotions = {'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
                           'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'}
        
        valences = []
        for text in texts:
            emotion = encoder.get_dominant_emotion(text)
            if emotion in positive_emotions:
                valences.append(1)
            elif emotion in negative_emotions:
                valences.append(-1)
            else:
                valences.append(0)
        
        # Consistency = 1 - normalized variance
        consistency = 1.0 - (np.var(valences) / 1.0)  # Max variance is 1.0
        return max(0.0, consistency)


# ==================== TESTING ====================

def test_text_encoder():
    """Test text encoder on sample texts"""
    encoder = TextEncoder()
    
    # Test texts
    test_texts = [
        "I am so happy and excited!",
        "This is terrible, I'm very disappointed.",
        "Everything is fine, just a normal day.",
        ""  # Empty text
    ]
    
    print("\n✓ Testing text encoder:\n")
    
    for text in test_texts:
        if text:
            features = encoder.extract_from_text(text)
            emotion = encoder.get_dominant_emotion(text)
            intensity = encoder.get_emotional_intensity(text)
            
            print(f"Text: '{text}'")
            print(f"  Feature shape: {features.shape}")
            print(f"  Dominant emotion: {emotion}")
            print(f"  Intensity: {intensity:.3f}\n")
        else:
            print("Empty text - returning zero vector\n")
    
    # Test batch processing
    features_batch = encoder.extract_from_texts(test_texts)
    print(f"✓ Batch features shape: {features_batch.shape}")


if __name__ == "__main__":
    test_text_encoder()