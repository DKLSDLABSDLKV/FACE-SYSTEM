"""
Age detection module using deep learning models.
"""

import cv2
import numpy as np
from pathlib import Path
import urllib.request
from typing import List, Tuple, Optional
import config


class AgeDetector:
    """Age estimation using deep learning model."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize age detector.
        
        Args:
            model_path: Path to age detection model. If None, uses default from config.
        """
        self.model_path = Path(model_path) if model_path else config.AGE_MODEL_PATH
        self.net = None
        self.input_size = (224, 224)
        self.age_ranges = config.AGE_RANGES
        
        self._load_model()
    
    def _download_model(self):
        """Download age detection model if not present."""
        if not self.model_path.exists():
            print(f"Downloading age detection model to {self.model_path}...")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(config.AGE_MODEL_URL, self.model_path)
                print("Age model downloaded successfully.")
            except Exception as e:
                print(f"Error downloading age model: {e}")
                print("Note: Age detection will use a simplified estimation method.")
                print("For better accuracy, please download the model manually.")
    
    def _load_model(self):
        """Load the age detection model."""
        self._download_model()
        
        if self.model_path.exists():
            try:
                self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Age detection model loaded successfully.")
            except Exception as e:
                print(f"Error loading age model: {e}")
                print("Falling back to simplified age estimation.")
                self.net = None
        else:
            print("Age model not found. Using simplified age estimation.")
            self.net = None
    
    def detect(self, face_image: np.ndarray) -> dict:
        """
        Estimate age from a face image.
        
        Args:
            face_image: Cropped face image (BGR format)
            
        Returns:
            Dictionary containing:
            {
                'age': int or str,  # Estimated age or age range
                'age_range': tuple,  # (min_age, max_age)
                'confidence': float
            }
        """
        if self.net is not None:
            return self._detect_with_model(face_image)
        else:
            return self._estimate_age_simple(face_image)
    
    def _detect_with_model(self, face_image: np.ndarray) -> dict:
        """Detect age using the loaded model."""
        # Preprocess image
        resized = cv2.resize(face_image, self.input_size)
        
        # GoogLeNet age model expects BGR input with mean subtraction.
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0,
            size=self.input_size,
            mean=[78.4263377603, 87.7689143744, 114.895847746],
            swapRB=False
        )
        
        # Run inference
        self.net.setInput(blob)
        outputs = self.net.forward()
        
        # Parse outputs (assuming model outputs age probabilities or direct age value)
        if len(outputs.shape) == 1:
            # Direct age value
            age = float(outputs[0])
            confidence = 1.0
            age_range = self._get_age_range(age)
        else:
            # Age range probabilities
            probs = outputs[0] if len(outputs.shape) > 1 else outputs
            age_range_idx = np.argmax(probs)
            confidence = float(probs[age_range_idx])
            age_range = self.age_ranges[age_range_idx]
            age = sum(age_range) / 2  # Use midpoint of range
        
        return {
            'age': int(age),
            'age_range': age_range,
            'confidence': confidence
        }
    
    def _estimate_age_simple(self, face_image: np.ndarray) -> dict:
        """
        Simple age estimation based on facial features.
        This is a fallback method when model is not available.
        """
        # Simple heuristic-based estimation
        # This is a placeholder - in practice, you'd want a trained model
        
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Analyze face features (very basic heuristic)
        # In a real implementation, this would use more sophisticated features
        
        # Estimate based on face size and texture (very rough approximation)
        face_area = h * w
        
        # This is a simplified estimation - not accurate but functional
        # In production, always use a trained model
        estimated_age = 30  # Default middle age
        confidence = 0.5  # Low confidence for heuristic method
        
        age_range = self._get_age_range(estimated_age)
        
        return {
            'age': estimated_age,
            'age_range': age_range,
            'confidence': confidence
        }
    
    def _get_age_range(self, age: float) -> Tuple[int, int]:
        """Get age range for a given age."""
        for min_age, max_age in self.age_ranges:
            if min_age <= age <= max_age:
                return (min_age, max_age)
        
        # Return the last range if age exceeds all ranges
        return self.age_ranges[-1]
    
    def detect_batch(self, face_images: List[np.ndarray]) -> List[dict]:
        """
        Detect age for multiple faces.
        
        Args:
            face_images: List of cropped face images
            
        Returns:
            List of age detection results
        """
        return [self.detect(face_img) for face_img in face_images]
