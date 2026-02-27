"""
Face detection module using OpenCV DNN.
"""

import cv2
import numpy as np
from pathlib import Path
import urllib.request
from typing import List, Tuple, Optional
import config


class FaceDetector:
    """Face detection using OpenCV FaceDetectorYN (YuNet) or DNN fallback."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize face detector.
        
        Args:
            model_path: Path to face detection model. If None, uses default from config.
        """
        self.model_path = Path(model_path) if model_path else config.FACE_DETECTOR_MODEL_PATH
        self.detector_yn = None  # cv2.FaceDetectorYN when available
        self.net = None  # Fallback: raw DNN
        self.input_size = (320, 320)
        self.score_threshold = config.FACE_DETECTION_THRESHOLD
        self.nms_threshold = 0.3
        self.top_k = 5000
        
        self._load_model()
    
    def _download_model(self):
        """Download face detection model if not present."""
        if not self.model_path.exists():
            print(f"Downloading face detection model to {self.model_path}...")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(config.FACE_DETECTOR_MODEL_URL, self.model_path)
                print("Model downloaded successfully.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("Please download the model manually or check your internet connection.")
                raise
    
    def _load_model(self):
        """Load the face detection model using FaceDetectorYN when available."""
        self._download_model()
        
        # Prefer OpenCV's FaceDetectorYN (handles preprocessing and output format)
        if hasattr(cv2, 'FaceDetectorYN'):
            try:
                self.detector_yn = cv2.FaceDetectorYN.create(
                    str(self.model_path),
                    "",
                    self.input_size,
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=self.top_k,
                )
                print("Face detection model loaded (FaceDetectorYN).")
                return
            except Exception as e:
                print(f"FaceDetectorYN failed: {e}, falling back to DNN.")
        
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Face detection model loaded (DNN).")
        except Exception as e:
            print(f"Error loading face detection model: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of dictionaries containing face detection results:
            {
                'bbox': [x, y, w, h],
                'confidence': float,
                'landmarks': [[x1, y1], [x2, y2], ...]  # 5 facial landmarks
            }
        """
        h, w = image.shape[:2]
        
        if self.detector_yn is not None:
            self.detector_yn.setInputSize((w, h))
            _, faces_mat = self.detector_yn.detect(image)
            if faces_mat is None:
                return []
            return self._parse_yn_faces(faces_mat)
        
        if self.net is None:
            raise RuntimeError("Model not loaded.")
        
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=self.input_size,
            mean=[104, 117, 123],
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()
        return self._parse_dnn_outputs(outputs, w, h)
    
    def _parse_yn_faces(self, faces_mat: np.ndarray) -> List[dict]:
        """Parse FaceDetectorYN output (N x 15)."""
        faces = []
        if faces_mat.size == 0:
            return faces
        if len(faces_mat.shape) == 1:
            faces_mat = faces_mat.reshape(1, -1)
        for i in range(faces_mat.shape[0]):
            d = faces_mat[i]
            if len(d) < 15:
                continue
            score = float(d[14])
            if score < self.score_threshold:
                continue
            x, y, width, height = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            landmarks = [
                [int(d[4]), int(d[5])],
                [int(d[6]), int(d[7])],
                [int(d[8]), int(d[9])],
                [int(d[10]), int(d[11])],
                [int(d[12]), int(d[13])],
            ]
            faces.append({
                "bbox": [x, y, width, height],
                "confidence": score,
                "landmarks": landmarks,
            })
        return faces
    
    def _parse_dnn_outputs(self, outputs: np.ndarray, img_w: int, img_h: int) -> List[dict]:
        """Parse raw DNN outputs (various shapes) into face list."""
        faces = []
        # Handle (1, N, 15) or (N, 15) or (1, 15, N)
        if len(outputs.shape) == 3:
            if outputs.shape[1] == 15 and outputs.shape[2] != 15:
                detections = outputs[0].T  # (15, N) -> (N, 15)
            else:
                detections = outputs[0]
        else:
            detections = np.atleast_2d(outputs)
        
        for detection in detections:
            if detection.size < 15:
                continue
            score = float(detection[14])
            if score < self.score_threshold:
                continue
            x = int(round(float(detection[0]) * img_w))
            y = int(round(float(detection[1]) * img_h))
            width = int(round(float(detection[2]) * img_w))
            height = int(round(float(detection[3]) * img_h))
            landmarks = [
                [int(round(float(detection[4]) * img_w)), int(round(float(detection[5]) * img_h))],
                [int(round(float(detection[6]) * img_w)), int(round(float(detection[7]) * img_h))],
                [int(round(float(detection[8]) * img_w)), int(round(float(detection[9]) * img_h))],
                [int(round(float(detection[10]) * img_w)), int(round(float(detection[11]) * img_h))],
                [int(round(float(detection[12]) * img_w)), int(round(float(detection[13]) * img_h))],
            ]
            faces.append({
                "bbox": [x, y, width, height],
                "confidence": score,
                "landmarks": landmarks,
            })
        if faces:
            faces = self._apply_nms(faces, img_w, img_h)
        return faces
    
    def _apply_nms(self, faces: List[dict], img_width: int, img_height: int) -> List[dict]:
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        if len(faces) == 0:
            return []
        boxes = [[f["bbox"][0], f["bbox"][1], f["bbox"][2], f["bbox"][3]] for f in faces]
        scores = [f["confidence"] for f in faces]
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.score_threshold, self.nms_threshold
        )
        if len(indices) == 0:
            return []
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        return [faces[i] for i in indices]
    
    def draw_detections(self, image: np.ndarray, faces: List[dict]) -> np.ndarray:
        """
        Draw face detections on image.
        
        Args:
            image: Input image
            faces: List of face detection results
            
        Returns:
            Image with drawn detections
        """
        result = image.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            confidence = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                config.BOX_COLOR,
                config.BOX_THICKNESS
            )
            
            # Draw confidence score
            label = f"Face: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                config.FONT_THICKNESS
            )
            cv2.rectangle(
                result,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                config.BOX_COLOR,
                -1
            )
            cv2.putText(
                result,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                (255, 255, 255),
                config.FONT_THICKNESS
            )
            
            # Draw landmarks
            for landmark in face['landmarks']:
                cv2.circle(result, tuple(landmark), 3, (0, 0, 255), -1)
        
        return result
