"""
Face Encoder - MediaPipe Face Landmarks
Extracts facial expressiveness features from video
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple
import torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import FACE_DIM, FACE_CONFIG


class FaceEncoder:
    """
    Frozen face encoder using MediaPipe Face Mesh
    Extracts 468 3D landmarks and computes expressiveness features
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize MediaPipe Face Mesh
        
        Args:
            config: Configuration dict (uses FACE_CONFIG if None)
        """
        self.config = config or FACE_CONFIG
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.config['max_faces'],
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence']
        )
        
        # Key landmark indices for expressiveness
        # Eyes, eyebrows, mouth, nose - most expressive regions
        self.key_landmarks = {
            'left_eye': [33, 160, 158, 133, 153, 144],
            'right_eye': [362, 385, 387, 263, 373, 380],
            'left_eyebrow': [70, 63, 105, 66, 107],
            'right_eyebrow': [336, 296, 334, 293, 300],
            'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
            'nose': [1, 2, 98, 327]
        }
        
        print("✓ Face Encoder initialized (MediaPipe)")
    
    def extract_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract face features from video file
        
        Args:
            video_path: Path to video file
        
        Returns:
            Face features array of shape (T, D) where T is frames, D is feature dim
            Returns None if no face detected
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        features_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract features from frame
            frame_features = self.extract_from_frame(rgb_frame)
            
            if frame_features is not None:
                features_sequence.append(frame_features)
        
        cap.release()
        
        if len(features_sequence) == 0:
            return None
        
        return np.array(features_sequence)
    
    def extract_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face features from single frame
        
        Args:
            frame: RGB image (H, W, 3)
        
        Returns:
            Feature vector of shape (D,) or None if no face detected
        """
        # Process frame
        results = self.face_mesh.process(frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face (we only process one face)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract landmark coordinates (468 landmarks × 2 coordinates = 936 features)
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y])
        
        landmarks = np.array(landmarks)
        
        # Compute additional expressiveness features
        expressiveness_features = self._compute_expressiveness(face_landmarks)
        
        # Combine: landmarks + expressiveness features
        features = np.concatenate([landmarks, expressiveness_features])
        
        return features
    
    def _compute_expressiveness(self, face_landmarks) -> np.ndarray:
        """
        Compute expressiveness features from landmarks
        - Eye openness
        - Mouth openness
        - Eyebrow position
        - Overall facial dynamics
        """
        landmarks_list = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
        
        features = []
        
        # Eye openness (left and right)
        left_eye_height = self._compute_distance(landmarks_list, 33, 160)
        right_eye_height = self._compute_distance(landmarks_list, 362, 385)
        features.extend([left_eye_height, right_eye_height])
        
        # Mouth openness (vertical)
        mouth_height = self._compute_distance(landmarks_list, 13, 14)
        features.append(mouth_height)
        
        # Mouth width
        mouth_width = self._compute_distance(landmarks_list, 61, 291)
        features.append(mouth_width)
        
        # Eyebrow position (distance from eyes)
        left_eyebrow_height = self._compute_distance(landmarks_list, 70, 145)
        right_eyebrow_height = self._compute_distance(landmarks_list, 300, 374)
        features.extend([left_eyebrow_height, right_eyebrow_height])
        
        # Facial asymmetry (left-right difference)
        left_cheek = landmarks_list[234]
        right_cheek = landmarks_list[454]
        asymmetry = abs(left_cheek[2] - right_cheek[2])  # Z-coordinate difference
        features.append(asymmetry)
        
        # Jaw openness
        jaw_openness = self._compute_distance(landmarks_list, 152, 10)
        features.append(jaw_openness)
        
        return np.array(features)
    
    @staticmethod
    def _compute_distance(landmarks: List[Tuple], idx1: int, idx2: int) -> float:
        """Compute Euclidean distance between two landmarks"""
        p1 = np.array(landmarks[idx1])
        p2 = np.array(landmarks[idx2])
        return np.linalg.norm(p1 - p2)
    
    def __call__(self, input_data):
        """
        Make encoder callable
        
        Args:
            input_data: Either video path (str) or frame (np.ndarray)
        
        Returns:
            Face features
        """
        if isinstance(input_data, str):
            return self.extract_from_video(input_data)
        elif isinstance(input_data, np.ndarray):
            return self.extract_from_frame(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def close(self):
        """Clean up resources"""
        self.face_mesh.close()


class FaceFeatureStatistics:
    """
    Compute temporal statistics for face features
    Used to assess regulation signals
    """
    
    @staticmethod
    def compute_variance(features: np.ndarray) -> float:
        """Compute temporal variance across sequence"""
        if len(features.shape) == 1:
            return 0.0
        return np.mean(np.var(features, axis=0))
    
    @staticmethod
    def compute_range(features: np.ndarray) -> float:
        """Compute temporal range (max - min)"""
        if len(features.shape) == 1:
            return 0.0
        return np.mean(np.ptp(features, axis=0))
    
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


# ==================== TESTING ====================

def test_face_encoder():
    """Test face encoder on sample video"""
    encoder = FaceEncoder()
    
    # Test on sample frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    features = encoder.extract_from_frame(dummy_frame)
    
    if features is not None:
        print(f"✓ Extracted features shape: {features.shape}")
        print(f"✓ Feature dimension: {len(features)}")
    else:
        print("⚠️  No face detected in test frame")
    
    encoder.close()


if __name__ == "__main__":
    test_face_encoder()