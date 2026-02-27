"""
Example: Single image processing.
"""

from detectors import FaceDetector, AgeDetector, EmotionDetector
from output_handler import DisplayOutput
import cv2
import sys


def main():
    """Run image processing example."""
    if len(sys.argv) < 2:
        print("Usage: python image_example.py <image_path> [--display]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    display_enabled = '--display' in sys.argv
    
    print("Initializing detectors...")
    face_detector = FaceDetector()
    age_detector = AgeDetector()
    emotion_detector = EmotionDetector()
    
    print(f"Loading image: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    print("Processing image...")
    
    # Detect faces
    faces = face_detector.detect(frame)
    
    # Process each face
    detections = {'faces': []}
    for face in faces:
        x, y, w, h = face['bbox']
        pad_h, pad_w = int(h * 0.2), int(w * 0.2)
        y1, y2 = max(0, y - pad_h), min(frame.shape[0], y + h + pad_h)
        x1, x2 = max(0, x - pad_w), min(frame.shape[1], x + w + pad_w)
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size > 0:
            age_result = age_detector.detect(face_roi)
            emotion_result = emotion_detector.detect(face_roi)
            
            detections['faces'].append({
                'bbox': face['bbox'],
                'confidence': face['confidence'],
                'age': age_result,
                'emotion': emotion_result
            })
    
    # Print results
    print(f"\nDetection Results:")
    print(f"  Faces detected: {len(detections['faces'])}")
    
    for i, face_info in enumerate(detections['faces']):
        print(f"\n  Face {i+1}:")
        print(f"    Bounding box: {face_info['bbox']}")
        print(f"    Confidence: {face_info['confidence']:.2f}")
        
        if 'age' in face_info:
            age_info = face_info['age']
            print(f"    Age: {age_info.get('age', 'N/A')}")
            print(f"    Age Range: {age_info.get('age_range', (0, 0))}")
            print(f"    Age Confidence: {age_info.get('confidence', 0.0):.2f}")
        
        if 'emotion' in face_info:
            emotion_info = face_info['emotion']
            print(f"    Emotion: {emotion_info.get('emotion', 'N/A')}")
            print(f"    Emotion Confidence: {emotion_info.get('confidence', 0.0):.2f}")
    
    # Display if requested
    if display_enabled:
        display = DisplayOutput()
        display.show(frame, detections)
        print("\nPress any key to close...")
        display.wait_key(0)
        display.destroy()


if __name__ == "__main__":
    main()
