"""
Audio Encoder - MFCC + Prosody Features
Extracts arousal and expressiveness signals from speech audio
"""

import librosa
import numpy as np
from typing import Optional, Dict
import soundfile as sf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import AUDIO_DIM, AUDIO_CONFIG


class AudioEncoder:
    """
    Frozen audio encoder using traditional signal processing
    Extracts MFCC, pitch, energy, and speaking rate
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize audio encoder
        
        Args:
            config: Configuration dict (uses AUDIO_CONFIG if None)
        """
        self.config = config or AUDIO_CONFIG
        self.sample_rate = self.config['sample_rate']
        self.n_mfcc = self.config['n_mfcc']
        self.n_fft = self.config['n_fft']
        self.hop_length = self.config['hop_length']
        
        print("✓ Audio Encoder initialized (MFCC + Prosody)")
    
    def extract_from_file(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract audio features from file
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)
        
        Returns:
            Audio features of shape (T, D) where T is frames, D is feature dim
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if len(y) == 0:
                return None
            
            return self.extract_from_signal(y, sr)
        
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def extract_from_signal(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract audio features from signal
        
        Args:
            y: Audio signal (1D array)
            sr: Sample rate (uses config if None)
        
        Returns:
            Audio features of shape (T, D)
        """
        if sr is None:
            sr = self.sample_rate
        
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
        
        # Extract all features
        features = []
        
        # 1. MFCC features (13 coefficients)
        mfcc = self._extract_mfcc(y)
        features.append(mfcc)
        
        # 2. Delta MFCC (13 coefficients)
        mfcc_delta = librosa.feature.delta(mfcc)
        features.append(mfcc_delta)
        
        # 3. Delta-delta MFCC (13 coefficients)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features.append(mfcc_delta2)
        
        # 4. Pitch (F0) - fundamental frequency
        pitch = self._extract_pitch(y)
        features.append(pitch.reshape(1, -1))
        
        # 5. Energy (RMS)
        energy = self._extract_energy(y)
        features.append(energy.reshape(1, -1))
        
        # 6. Zero Crossing Rate (speaking rate proxy)
        zcr = self._extract_zcr(y)
        features.append(zcr.reshape(1, -1))
        
        # 7. Spectral centroid (timbre/brightness)
        spectral_centroid = self._extract_spectral_centroid(y)
        features.append(spectral_centroid.reshape(1, -1))
        
        # 8. Spectral rolloff (energy distribution)
        spectral_rolloff = self._extract_spectral_rolloff(y)
        features.append(spectral_rolloff.reshape(1, -1))
        
        # Concatenate all features
        # Shape: (n_features, n_frames)
        combined = np.vstack(features)
        
        # Transpose to (n_frames, n_features)
        combined = combined.T
        
        return combined
    
    def _extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc
    
    def _extract_pitch(self, y: np.ndarray) -> np.ndarray:
        """
        Extract pitch (F0) using librosa.pyin
        Returns fundamental frequency over time
        """
        # Use PYIN algorithm for pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Replace NaN with 0 (unvoiced segments)
        f0 = np.nan_to_num(f0, nan=0.0)
        
        return f0
    
    def _extract_energy(self, y: np.ndarray) -> np.ndarray:
        """Extract RMS energy"""
        energy = librosa.feature.rms(
            y=y,
            hop_length=self.hop_length
        )[0]
        return energy
    
    def _extract_zcr(self, y: np.ndarray) -> np.ndarray:
        """Extract zero crossing rate (speaking rate indicator)"""
        zcr = librosa.feature.zero_crossing_rate(
            y,
            hop_length=self.hop_length
        )[0]
        return zcr
    
    def _extract_spectral_centroid(self, y: np.ndarray) -> np.ndarray:
        """Extract spectral centroid (timbre/brightness)"""
        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )[0]
        return centroid
    
    def _extract_spectral_rolloff(self, y: np.ndarray) -> np.ndarray:
        """Extract spectral rolloff (energy distribution)"""
        rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )[0]
        return rolloff
    
    def compute_prosody_statistics(self, features: np.ndarray) -> Dict[str, float]:
        """
        Compute prosody statistics from audio features
        Used for regulation analysis
        
        Args:
            features: Audio features (T, D)
        
        Returns:
            Dictionary with prosody statistics
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        stats = {}
        
        # Pitch statistics (feature index depends on extraction order)
        # MFCC (13) + delta (13) + delta2 (13) = 39, pitch is index 39
        if features.shape[1] > 39:
            pitch = features[:, 39]
            
            # Remove zeros (unvoiced)
            voiced_pitch = pitch[pitch > 0]
            
            if len(voiced_pitch) > 0:
                stats['pitch_mean'] = np.mean(voiced_pitch)
                stats['pitch_std'] = np.std(voiced_pitch)
                stats['pitch_range'] = np.ptp(voiced_pitch)
                stats['pitch_variance'] = np.var(voiced_pitch)
            else:
                stats['pitch_mean'] = 0.0
                stats['pitch_std'] = 0.0
                stats['pitch_range'] = 0.0
                stats['pitch_variance'] = 0.0
        
        # Energy statistics
        if features.shape[1] > 40:
            energy = features[:, 40]
            stats['energy_mean'] = np.mean(energy)
            stats['energy_std'] = np.std(energy)
            stats['energy_range'] = np.ptp(energy)
        
        # Speaking rate (zero crossing rate)
        if features.shape[1] > 41:
            zcr = features[:, 41]
            stats['speaking_rate'] = np.mean(zcr)
        
        # Overall variance (regulation indicator)
        stats['temporal_variance'] = np.mean(np.var(features, axis=0))
        
        return stats
    
    def __call__(self, input_data):
        """
        Make encoder callable
        
        Args:
            input_data: Either audio path (str) or signal (np.ndarray)
        
        Returns:
            Audio features
        """
        if isinstance(input_data, str):
            return self.extract_from_file(input_data)
        elif isinstance(input_data, np.ndarray):
            return self.extract_from_signal(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")


class AudioFeatureStatistics:
    """
    Compute temporal statistics for audio features
    """
    
    @staticmethod
    def compute_variance(features: np.ndarray) -> float:
        """Compute temporal variance"""
        if len(features.shape) == 1:
            return 0.0
        return np.mean(np.var(features, axis=0))
    
    @staticmethod
    def compute_smoothness(features: np.ndarray) -> float:
        """
        Compute smoothness via frame-to-frame differences
        Lower = smoother = more controlled
        """
        if len(features) < 2:
            return 0.0
        
        diffs = np.diff(features, axis=0)
        smoothness = np.mean(np.linalg.norm(diffs, axis=1))
        return smoothness
    
    @staticmethod
    def compute_dynamics_range(features: np.ndarray) -> float:
        """Compute dynamic range (max - min)"""
        if len(features.shape) == 1:
            return 0.0
        return np.mean(np.ptp(features, axis=0))


# ==================== TESTING ====================

def test_audio_encoder():
    """Test audio encoder on sample audio"""
    encoder = AudioEncoder()
    
    # Generate test audio (1 second of sine wave)
    duration = 1.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    features = encoder.extract_from_signal(y, sr)
    
    print(f"✓ Extracted features shape: {features.shape}")
    print(f"✓ Feature dimension per frame: {features.shape[1]}")
    
    # Test prosody statistics
    prosody = encoder.compute_prosody_statistics(features)
    print("\n✓ Prosody statistics:")
    for key, value in prosody.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    test_audio_encoder()