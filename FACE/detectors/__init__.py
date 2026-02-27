"""
Detection modules for age, emotion, motion, and face detection.
"""

from .face_detector import FaceDetector
from .age_detector import AgeDetector
from .emotion_detector import EmotionDetector
from .motion_detector import MotionDetector

__all__ = ['FaceDetector', 'AgeDetector', 'EmotionDetector', 'MotionDetector']
