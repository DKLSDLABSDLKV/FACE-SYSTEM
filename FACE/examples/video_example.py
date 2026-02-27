"""
Example: Video file processing with results export.
"""

from detectors import FaceDetector, AgeDetector, EmotionDetector, MotionDetector
from input_handler import VideoInput
from output_handler import FileOutput
import sys


def main():
    """Run video processing example."""
    if len(sys.argv) < 2:
        print("Usage: python video_example.py <video_path> [output_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "video_results.json"
    
    print("Initializing detectors...")
    face_detector = FaceDetector()
    age_detector = AgeDetector()
    emotion_detector = EmotionDetector()
    motion_detector = MotionDetector()
    
    print(f"Loading video: {video_path}")
    video = VideoInput(video_path)
    
    file_output = FileOutput(output_path, format="json")
    
    print("Processing video...")
    frame_number = 0
    
    try:
        while True:
            frame, has_more = video.get_frame()
            
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
            
            # Save result
            file_output.add_result(frame_number, detections)
            
            frame_number += 1
            
            if frame_number % 30 == 0:
                print(f"Processed {frame_number} frames...")
            
            if not has_more:
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        file_output.save()
        video.release()
        print(f"Processed {frame_number} frames. Results saved to {output_path}")


if __name__ == "__main__":
    main()
