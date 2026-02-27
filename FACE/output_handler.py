"""
Output handler for display, file export, and API integration.
"""

import cv2
import numpy as np
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import config


class DisplayOutput:
    """Real-time display output with annotations."""
    
    def __init__(self, window_name: str = "Detection Output", width: int = None, height: int = None):
        """
        Initialize display output.
        
        Args:
            window_name: Name of the display window
            width: Display width (None for auto)
            height: Display height (None for auto)
        """
        self.window_name = window_name
        self.width = width or config.DISPLAY_WIDTH
        self.height = height or config.DISPLAY_HEIGHT
        self.frame_count = 0
        self.start_time = datetime.now()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
    
    def show(self, frame: np.ndarray, detections: Dict[str, Any] = None):
        """
        Display frame with annotations.
        
        Args:
            frame: Frame to display
            detections: Dictionary containing detection results
        """
        display_frame = frame.copy()
        
        if detections:
            display_frame = self._draw_detections(display_frame, detections)
        
        # Add FPS counter
        display_frame = self._draw_fps(display_frame)
        
        # Resize if needed
        h, w = display_frame.shape[:2]
        if w > self.width or h > self.height:
            scale = min(self.width / w, self.height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_frame = cv2.resize(display_frame, (new_w, new_h))
        
        cv2.imshow(self.window_name, display_frame)
        self.frame_count += 1
    
    def _draw_detections(self, frame: np.ndarray, detections: Dict[str, Any]) -> np.ndarray:
        """Draw detection annotations on frame."""
        result = frame.copy()
        
        # Draw face detections with age and emotion
        if 'faces' in detections:
            for face_info in detections['faces']:
                if 'bbox' in face_info:
                    x, y, w, h = face_info['bbox']
                    
                    # Draw bounding box
                    cv2.rectangle(result, (x, y), (x + w, y + h), config.BOX_COLOR, config.BOX_THICKNESS)
                    
                    # Prepare labels
                    labels = []
                    if 'age' in face_info:
                        age_info = face_info['age']
                        if isinstance(age_info, dict):
                            age = age_info.get('age', 'N/A')
                            age_range = age_info.get('age_range', (0, 0))
                            labels.append(f"Age: {age} ({age_range[0]}-{age_range[1]})")
                    
                    if 'emotion' in face_info:
                        emotion_info = face_info['emotion']
                        if isinstance(emotion_info, dict):
                            emotion = emotion_info.get('emotion', 'N/A')
                            confidence = emotion_info.get('confidence', 0.0)
                            labels.append(f"{emotion}: {confidence:.2f}")
                    
                    # Draw labels
                    y_offset = y - 10
                    for label in reversed(labels):
                        label_size, _ = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            config.FONT_SCALE,
                            config.FONT_THICKNESS
                        )
                        cv2.rectangle(
                            result,
                            (x, y_offset - label_size[1] - 5),
                            (x + label_size[0], y_offset),
                            config.BOX_COLOR,
                            -1
                        )
                        cv2.putText(
                            result,
                            label,
                            (x, y_offset - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            config.FONT_SCALE,
                            (255, 255, 255),
                            config.FONT_THICKNESS
                        )
                        y_offset -= label_size[1] + 10
        
        # Draw motion regions
        if 'motion' in detections and detections['motion'].get('motion_detected', False):
            motion_info = detections['motion']
            if 'motion_regions' in motion_info:
                for region in motion_info['motion_regions']:
                    if 'bbox' in region:
                        x, y, w, h = region['bbox']
                        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Draw motion magnitude
            magnitude = motion_info.get('motion_magnitude', 0.0)
            cv2.putText(
                result,
                f"Motion: {magnitude:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                (0, 0, 255),
                config.FONT_THICKNESS
            )
        
        return result
    
    def _draw_fps(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS counter on frame."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > 0:
            fps = self.frame_count / elapsed
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                config.FONT_COLOR,
                config.FONT_THICKNESS
            )
        return frame
    
    def wait_key(self, delay: int = 1) -> int:
        """Wait for key press."""
        return cv2.waitKey(delay) & 0xFF
    
    def destroy(self):
        """Destroy display window."""
        cv2.destroyWindow(self.window_name)


class FileOutput:
    """File output for saving detection results."""
    
    def __init__(self, output_path: str, format: str = "json"):
        """
        Initialize file output.
        
        Args:
            output_path: Path to output file
            format: Output format - "json" or "csv"
        """
        self.output_path = Path(output_path)
        self.format = format.lower()
        self.results = []
        
        if self.format not in ["json", "csv"]:
            raise ValueError(f"Unsupported format: {format}. Must be 'json' or 'csv'")
    
    def add_result(self, frame_number: int, detections: Dict[str, Any]):
        """
        Add detection result for a frame.
        
        Args:
            frame_number: Frame number
            detections: Detection results
        """
        result = {
            'frame_number': frame_number,
            'timestamp': datetime.now().isoformat(),
            'detections': detections
        }
        self.results.append(result)
    
    def save(self):
        """Save all results to file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.format == "json":
            self._save_json()
        elif self.format == "csv":
            self._save_csv()
    
    def _save_json(self):
        """Save results as JSON."""
        output_data = {
            'metadata': {
                'total_frames': len(self.results),
                'exported_at': datetime.now().isoformat()
            },
            'results': self.results
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=config.OUTPUT_JSON_INDENT)
        
        print(f"Results saved to {self.output_path}")
    
    def _save_csv(self):
        """Save results as CSV."""
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=config.OUTPUT_CSV_DELIMITER)
            
            # Write header
            writer.writerow([
                'frame_number',
                'timestamp',
                'num_faces',
                'motion_detected',
                'motion_magnitude'
            ])
            
            # Write data rows
            for result in self.results:
                detections = result['detections']
                num_faces = len(detections.get('faces', []))
                motion_detected = detections.get('motion', {}).get('motion_detected', False)
                motion_magnitude = detections.get('motion', {}).get('motion_magnitude', 0.0)
                
                writer.writerow([
                    result['frame_number'],
                    result['timestamp'],
                    num_faces,
                    motion_detected,
                    motion_magnitude
                ])
        
        print(f"Results saved to {self.output_path}")


class APIOutput:
    """API output formatter for REST API responses."""
    
    @staticmethod
    def format_detection_result(detections: Dict[str, Any], frame_number: int = None) -> Dict[str, Any]:
        """
        Format detection results for API response.
        
        Args:
            detections: Detection results
            frame_number: Optional frame number
            
        Returns:
            Formatted dictionary for JSON response
        """
        result = {
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
        
        if frame_number is not None:
            result['frame_number'] = frame_number
        
        # Format faces
        if 'faces' in detections:
            result['faces'] = []
            for face_info in detections['faces']:
                face_data = {
                    'bbox': face_info.get('bbox'),
                    'confidence': face_info.get('confidence', 0.0)
                }
                
                if 'age' in face_info:
                    face_data['age'] = face_info['age']
                
                if 'emotion' in face_info:
                    face_data['emotion'] = face_info['emotion']
                
                result['faces'].append(face_data)
        
        # Format motion
        if 'motion' in detections:
            motion_info = detections['motion']
            result['motion'] = {
                'detected': motion_info.get('motion_detected', False),
                'magnitude': motion_info.get('motion_magnitude', 0.0),
                'regions': motion_info.get('motion_regions', []),
                'statistics': motion_info.get('statistics', {})
            }
        
        return result
    
    @staticmethod
    def format_error(error_message: str) -> Dict[str, Any]:
        """
        Format error response.
        
        Args:
            error_message: Error message
            
        Returns:
            Formatted error dictionary
        """
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }
