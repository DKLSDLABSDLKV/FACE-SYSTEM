"""
Example: Webcam detection with real-time display.
"""

from detectors import FaceDetector, AgeDetector, EmotionDetector, MotionDetector
from input_handler import WebcamInput
from output_handler import DisplayOutput
import cv2
import config


def main():
    """Run webcam detection example."""
    print("Initializing detectors...")
    face_detector = FaceDetector()
    age_detector = AgeDetector()
    emotion_detector = EmotionDetector()
    motion_detector = MotionDetector()
    
    print("Opening webcam...")
    webcam = WebcamInput(camera_index=config.WEBCAM_INDEX)
    display = DisplayOutput()
    
    print("Starting detection. Press 'q' to quit.")
    
    try:
        while True:
            frame, has_more = webcam.get_frame()
            
            if frame is None:
                break
            
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
            
            # Detect motion
            motion_result = motion_detector.detect(frame)
            detections['motion'] = motion_result
            
            # Display
            display.show(frame, detections)
            
            # Check for quit
            if display.wait_key(1) == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        display.destroy()
        webcam.release()
        print("Done.")


if __name__ == "__main__":
    main()
