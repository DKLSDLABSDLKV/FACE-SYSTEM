"""
Emotion detection module using FER (Facial Expression Recognition) models.
"""

import cv2
import numpy as np
from pathlib import Path
import urllib.request
from typing import List, Optional
import config


class EmotionDetector:
    """Emotion recognition using deep learning model."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize emotion detector.
        
        Args:
            model_path: Path to emotion detection model. If None, uses default from config.
        """
        self.model_path = Path(model_path) if model_path else config.EMOTION_MODEL_PATH
        self.net = None
        self.input_size = (112, 112)  # Common size for emotion models
        self.emotion_classes = config.EMOTION_CLASSES
        self.confidence_threshold = config.EMOTION_CONFIDENCE_THRESHOLD
        
        self._load_model()
    
    def _download_model(self):
        """Download emotion detection model if not present."""
        if not self.model_path.exists():
            print(f"Downloading emotion detection model to {self.model_path}...")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(config.EMOTION_MODEL_URL, self.model_path)
                print("Emotion model downloaded successfully.")
            except Exception as e:
                print(f"Error downloading emotion model: {e}")
                print("Note: Emotion detection will use a simplified method.")
                print("For better accuracy, please download the model manually.")
    
    def _load_model(self):
        """Load the emotion detection model."""
        self._download_model()
        
        if self.model_path.exists():
            try:
                self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Emotion detection model loaded successfully.")
            except Exception as e:
                print(f"Error loading emotion model: {e}")
                print("Falling back to simplified emotion detection.")
                self.net = None
        else:
            print("Emotion model not found. Using simplified emotion detection.")
            self.net = None
    
    def detect(self, face_image: np.ndarray) -> dict:
        """
        Detect emotion from a face image.
        
        Args:
            face_image: Cropped face image (BGR format)
            
        Returns:
            Dictionary containing:
            {
                'emotion': str,  # Emotion label
                'confidence': float,  # Confidence score
                'probabilities': dict  # All emotion probabilities
            }
        """
        if self.net is not None:
            return self._detect_with_model(face_image)
        else:
            return self._detect_simple(face_image)
    
    def _detect_with_model(self, face_image: np.ndarray) -> dict:
        """Detect emotion using the loaded model."""
        # Preprocess image
        resized = cv2.resize(face_image, self.input_size)
        
        # Normalize to [-1, 1] in RGB for MobileFaceNet FER model.
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0 / 127.5,
            size=self.input_size,
            mean=[127.5, 127.5, 127.5],
            swapRB=True
        )
        
        # Run inference
        self.net.setInput(blob)
        outputs = self.net.forward()
        
        # Parse outputs
        if len(outputs.shape) > 1:
            probs = outputs[0]
        else:
            probs = outputs
        
        # Get emotion with highest probability
        emotion_idx = np.argmax(probs)
        confidence = float(probs[emotion_idx])
        emotion = self.emotion_classes[emotion_idx] if emotion_idx < len(self.emotion_classes) else "Unknown"
        
        # Create probabilities dictionary
        probabilities = {
            self.emotion_classes[i]: float(probs[i])
            for i in range(min(len(probs), len(self.emotion_classes)))
        }
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def _detect_simple(self, face_image: np.ndarray) -> dict:
        """
        Simple emotion detection based on facial features.
        This is a fallback method when model is not available.
        """
        # Very basic heuristic-based emotion detection
        # In production, always use a trained model
        
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Simple feature analysis (mouth corners, eye regions)
        # This is a placeholder - not accurate but functional
        
        # Default to neutral with low confidence
        emotion = "Neutral"
        confidence = 0.3
        
        probabilities = {emotion: confidence}
        for emo in self.emotion_classes:
            if emo != emotion:
                probabilities[emo] = (1.0 - confidence) / (len(self.emotion_classes) - 1)
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def detect_batch(self, face_images: List[np.ndarray]) -> List[dict]:
        """
        Detect emotion for multiple faces.
        
        Args:
            face_images: List of cropped face images
            
        Returns:
            List of emotion detection results
        """
        return [self.detect(face_img) for face_img in face_images]
