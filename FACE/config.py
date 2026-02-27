"""
Configuration settings for the detection system.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Models directory
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Face detection model
FACE_DETECTOR_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
FACE_DETECTOR_MODEL_PATH = MODELS_DIR / "face_detection_yunet.onnx"

# Age detection model
AGE_MODEL_DIR = MODELS_DIR / "age_model"
AGE_MODEL_DIR.mkdir(exist_ok=True)
AGE_MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/age_gender/models/age_googlenet.onnx"
AGE_MODEL_PATH = AGE_MODEL_DIR / "age_recognition.onnx"

# Emotion detection model
EMOTION_MODEL_DIR = MODELS_DIR / "emotion_model"
EMOTION_MODEL_DIR.mkdir(exist_ok=True)
EMOTION_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx"
EMOTION_MODEL_PATH = EMOTION_MODEL_DIR / "facial_expression_recognition.onnx"

# Detection thresholds
FACE_DETECTION_THRESHOLD = 0.8
AGE_CONFIDENCE_THRESHOLD = 0.5
EMOTION_CONFIDENCE_THRESHOLD = 0.5
MOTION_THRESHOLD = 30.0  # Minimum motion magnitude

# Age ranges
AGE_RANGES = [
    (0, 2),
    (4, 6),
    (8, 12),
    (15, 20),
    (25, 32),
    (38, 43),
    (48, 53),
    (60, 100)
]

# Emotion classes
EMOTION_CLASSES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Display settings
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
FONT_SCALE = 0.6
FONT_THICKNESS = 2
FONT_COLOR = (0, 255, 0)
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Age, Emotion, Motion Detection API"
API_VERSION = "1.0.0"

# Frame processing settings
FRAME_SKIP = 1  # Process every Nth frame (1 = process all frames)
MAX_FRAME_SIZE = (1920, 1080)  # Maximum frame size for processing

# Motion detection settings
MOTION_BACKGROUND_HISTORY = 500
MOTION_VAR_THRESHOLD = 16
MOTION_DETECT_SHADOWS = True
OPTICAL_FLOW_WINDOW_SIZE = 15

# Webcam settings
WEBCAM_INDEX = 0
WEBCAM_FPS = 30

# Output settings
OUTPUT_JSON_INDENT = 2
OUTPUT_CSV_DELIMITER = ","
