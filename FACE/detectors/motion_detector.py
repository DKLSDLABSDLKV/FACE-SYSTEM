"""
Motion and movement detection module using optical flow and background subtraction.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import config


class MotionDetector:
    """Motion and movement detection using optical flow and background subtraction."""
    
    def __init__(self):
        """Initialize motion detector."""
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOTION_BACKGROUND_HISTORY,
            varThreshold=config.MOTION_VAR_THRESHOLD,
            detectShadows=config.MOTION_DETECT_SHADOWS
        )
        
        # Previous frame for optical flow
        self.prev_gray = None
        self.prev_points = None
        
        # Motion threshold
        self.motion_threshold = config.MOTION_THRESHOLD
        
        # Optical flow parameters
        self.optical_flow_params = dict(
            winSize=(config.OPTICAL_FLOW_WINDOW_SIZE, config.OPTICAL_FLOW_WINDOW_SIZE),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def detect(self, frame: np.ndarray, method: str = "both") -> dict:
        """
        Detect motion in a frame.
        
        Args:
            frame: Input frame (BGR format)
            method: Detection method - "background", "optical_flow", or "both"
            
        Returns:
            Dictionary containing:
            {
                'motion_detected': bool,
                'motion_magnitude': float,
                'motion_heatmap': np.ndarray,  # Motion visualization
                'motion_regions': list,  # List of bounding boxes with motion
                'statistics': dict  # Motion statistics
            }
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        # Use 3-channel heatmap so we can add color heatmaps from background/optical flow
        heatmap_shape = (h, w, 3)
        results = {
            'motion_detected': False,
            'motion_magnitude': 0.0,
            'motion_heatmap': np.zeros(heatmap_shape, dtype=np.uint8),
            'motion_regions': [],
            'statistics': {}
        }
        
        if method in ["background", "both"]:
            bg_result = self._detect_background_subtraction(gray)
            results['motion_detected'] = bg_result['motion_detected']
            results['motion_magnitude'] = max(results['motion_magnitude'], bg_result['motion_magnitude'])
            results['motion_heatmap'] = bg_result['motion_heatmap'].copy()
            results['motion_regions'].extend(bg_result['motion_regions'])
            results['statistics']['background'] = bg_result['statistics']
        
        if method in ["optical_flow", "both"]:
            flow_result = self._detect_optical_flow(gray)
            if flow_result['motion_detected']:
                results['motion_detected'] = True
            results['motion_magnitude'] = max(results['motion_magnitude'], flow_result['motion_magnitude'])
            # Both heatmaps are (H,W,3); use one or blend when both are used
            if method == "both" and results['motion_heatmap'].size > 0 and flow_result['motion_heatmap'].size > 0:
                results['motion_heatmap'] = cv2.addWeighted(results['motion_heatmap'], 0.5, flow_result['motion_heatmap'], 0.5, 0)
            elif method == "optical_flow":
                results['motion_heatmap'] = flow_result['motion_heatmap'].copy()
            results['motion_regions'].extend(flow_result['motion_regions'])
            results['statistics']['optical_flow'] = flow_result['statistics']
        
        # Update previous frame
        self.prev_gray = gray.copy()
        
        return results
    
    def _detect_background_subtraction(self, gray: np.ndarray) -> dict:
        """Detect motion using background subtraction."""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate motion magnitude
        motion_pixels = np.sum(fg_mask > 0)
        total_pixels = gray.shape[0] * gray.shape[1]
        motion_magnitude = (motion_pixels / total_pixels) * 100.0
        
        motion_detected = motion_magnitude > self.motion_threshold
        
        # Find motion regions (contours)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append({
                    'bbox': [x, y, w, h],
                    'area': area
                })
        
        # Create heatmap
        heatmap = cv2.applyColorMap(fg_mask, cv2.COLORMAP_HOT)
        
        statistics = {
            'motion_pixels': int(motion_pixels),
            'total_pixels': int(total_pixels),
            'motion_percentage': motion_magnitude,
            'num_regions': len(motion_regions)
        }
        
        return {
            'motion_detected': motion_detected,
            'motion_magnitude': motion_magnitude,
            'motion_heatmap': heatmap,
            'motion_regions': motion_regions,
            'statistics': statistics
        }
    
    def _detect_optical_flow(self, gray: np.ndarray) -> dict:
        """Detect motion using optical flow."""
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            h, w = gray.shape[:2]
            return {
                'motion_detected': False,
                'motion_magnitude': 0.0,
                'motion_heatmap': np.zeros((h, w, 3), dtype=np.uint8),
                'motion_regions': [],
                'statistics': {}
            }
        
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5,  # pyr_scale
            3,    # levels
            15,   # winsize
            3,    # iterations
            5,    # poly_n
            1.2,  # poly_sigma
            0     # flags
        )
        
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Filter by magnitude threshold
        motion_mask = magnitude > self.motion_threshold
        
        # Calculate motion statistics
        motion_magnitude = np.mean(magnitude)
        motion_detected = motion_magnitude > self.motion_threshold
        
        # Create heatmap
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(magnitude_normalized, cv2.COLORMAP_JET)
        
        # Find motion regions
        motion_regions = []
        if motion_detected:
            # Create binary mask from motion
            motion_binary = (magnitude > self.motion_threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(motion_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_regions.append({
                        'bbox': [x, y, w, h],
                        'area': area,
                        'avg_magnitude': float(np.mean(magnitude[y:y+h, x:x+w]))
                    })
        
        statistics = {
            'avg_magnitude': float(motion_magnitude),
            'max_magnitude': float(np.max(magnitude)),
            'num_regions': len(motion_regions)
        }
        
        return {
            'motion_detected': motion_detected,
            'motion_magnitude': float(motion_magnitude),
            'motion_heatmap': heatmap,
            'motion_regions': motion_regions,
            'statistics': statistics
        }
    
    def reset(self):
        """Reset the background subtractor and previous frame."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOTION_BACKGROUND_HISTORY,
            varThreshold=config.MOTION_VAR_THRESHOLD,
            detectShadows=config.MOTION_DETECT_SHADOWS
        )
        self.prev_gray = None
        self.prev_points = None
