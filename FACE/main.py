"""
Main application for age, emotion, and motion detection system.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys
import time

from detectors import FaceDetector, AgeDetector, EmotionDetector, MotionDetector
from input_handler import create_input_handler
from output_handler import DisplayOutput, FileOutput
import config


class DetectionSystem:
    """Main detection system orchestrator."""
    
    def __init__(self):
        """Initialize detection system with all detectors."""
        print("Initializing detection system...")
        
        self.face_detector = FaceDetector()
        self.age_detector = AgeDetector()
        self.emotion_detector = EmotionDetector()
        self.motion_detector = MotionDetector()
        
        print("All detectors initialized successfully.")
    
    def process_frame(self, frame: np.ndarray, detect_motion: bool = True) -> dict:
        """
        Process a single frame and return detection results.
        
        Args:
            frame: Input frame (BGR format)
            detect_motion: Whether to detect motion
            
        Returns:
            Dictionary containing all detection results
        """
        results = {
            'faces': [],
            'motion': {}
        }
        
        # Detect faces
        faces = self.face_detector.detect(frame)
        
        # Process each face for age and emotion
        for face in faces:
            x, y, w, h = face['bbox']
            pad_h, pad_w = int(h * 0.2), int(w * 0.2)
            y1, y2 = max(0, y - pad_h), min(frame.shape[0], y + h + pad_h)
            x1, x2 = max(0, x - pad_w), min(frame.shape[1], x + w + pad_w)
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size > 0:
                # Detect age
                age_result = self.age_detector.detect(face_roi)
                
                # Detect emotion
                emotion_result = self.emotion_detector.detect(face_roi)
                
                # Combine results
                face_info = {
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'landmarks': face['landmarks'],
                    'age': age_result,
                    'emotion': emotion_result
                }
                
                results['faces'].append(face_info)
        
        # Detect motion
        if detect_motion:
            motion_result = self.motion_detector.detect(frame)
            results['motion'] = motion_result
        
        return results
    
    def process_video(self, input_handler, display: bool = False, output_path: str = None):
        """
        Process video/webcam stream.
        
        Args:
            input_handler: Input handler instance
            display: Whether to display results
            output_path: Optional path to save results
        """
        display_output = None
        file_output = None
        
        if display:
            display_output = DisplayOutput()
        
        if output_path:
            file_format = Path(output_path).suffix[1:] if Path(output_path).suffix else "json"
            file_output = FileOutput(output_path, format=file_format)
        
        frame_number = 0
        frame_skip = config.FRAME_SKIP
        
        try:
            print("Starting processing...")
            print("Press 'q' to quit, 's' to save screenshot")
            
            while True:
                frame, has_more = input_handler.get_frame()
                
                if frame is None:
                    break
                
                # Skip frames if configured
                if frame_number % frame_skip != 0:
                    frame_number += 1
                    continue
                
                # Resize frame if too large
                h, w = frame.shape[:2]
                max_w, max_h = config.MAX_FRAME_SIZE
                if w > max_w or h > max_h:
                    scale = min(max_w / w, max_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))
                
                # Process frame
                detections = self.process_frame(frame)
                
                # Display
                if display_output:
                    display_output.show(frame, detections)
                
                # Save to file
                if file_output:
                    file_output.add_result(frame_number, detections)
                
                frame_number += 1
                
                # Check for key press
                if display_output:
                    key = display_output.wait_key(1)
                    if key == ord('q'):
                        print("Quitting...")
                        break
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_{frame_number}.jpg"
                        cv2.imwrite(screenshot_path, frame)
                        print(f"Screenshot saved: {screenshot_path}")
                
                if not has_more:
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            if display_output:
                display_output.destroy()
            
            if file_output:
                file_output.save()
            
            input_handler.release()
            
            print(f"Processed {frame_number} frames")
    
    def process_image(self, image_path: str, output_path: str = None, display: bool = False):
        """
        Process a single image.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save results
            display: Whether to display results
        """
        print(f"Processing image: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Process frame
        detections = self.process_frame(frame, detect_motion=False)
        
        # Display
        if display:
            display_output = DisplayOutput()
            display_output.show(frame, detections)
            print("Press any key to close...")
            display_output.wait_key(0)
            display_output.destroy()
        
        # Save results
        if output_path:
            file_format = Path(output_path).suffix[1:] if Path(output_path).suffix else "json"
            file_output = FileOutput(output_path, format=file_format)
            file_output.add_result(0, detections)
            file_output.save()
        
        # Print summary
        print(f"\nDetection Results:")
        print(f"  Faces detected: {len(detections['faces'])}")
        for i, face in enumerate(detections['faces']):
            age_info = face.get('age', {})
            emotion_info = face.get('emotion', {})
            print(f"  Face {i+1}:")
            if age_info:
                print(f"    Age: {age_info.get('age', 'N/A')} ({age_info.get('age_range', (0,0))[0]}-{age_info.get('age_range', (0,0))[1]})")
            if emotion_info:
                print(f"    Emotion: {emotion_info.get('emotion', 'N/A')} ({emotion_info.get('confidence', 0.0):.2f})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Age, Emotion, and Motion Detection System"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['webcam', 'video', 'image'],
        default='webcam',
        help='Input mode: webcam, video, or image (default: webcam)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input path (required for video/image mode)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for saving results (JSON or CSV)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not open display window (e.g. when only saving to file)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=config.WEBCAM_INDEX,
        help=f'Camera index (default: {config.WEBCAM_INDEX})'
    )
    
    args = parser.parse_args()
    
    # Display on by default unless --no-display
    display = not args.no_display
    
    # Validate arguments
    if args.mode in ['video', 'image'] and not args.input:
        parser.error(f"--input is required for {args.mode} mode")
    
    # Initialize detection system
    try:
        system = DetectionSystem()
    except Exception as e:
        print(f"Error initializing detection system: {e}")
        sys.exit(1)
    
    # Process based on mode
    try:
        if args.mode == 'webcam':
            input_handler = create_input_handler('webcam', camera_index=args.camera)
            system.process_video(input_handler, display=display, output_path=args.output)
        
        elif args.mode == 'video':
            input_handler = create_input_handler('video', input_path=args.input)
            system.process_video(input_handler, display=display, output_path=args.output)
        
        elif args.mode == 'image':
            system.process_image(args.input, output_path=args.output, display=display)
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
