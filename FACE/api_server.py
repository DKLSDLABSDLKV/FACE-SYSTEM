"""
FastAPI REST API server for detection system.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
from typing import Optional
import uvicorn
import asyncio
import base64

from detectors import FaceDetector, AgeDetector, EmotionDetector, MotionDetector
from output_handler import APIOutput
import config


# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detectors (lazy loading)
detectors_initialized = False
face_detector = None
age_detector = None
emotion_detector = None
motion_detector = None


def initialize_detectors():
    """Initialize all detectors (called on first request)."""
    global detectors_initialized, face_detector, age_detector, emotion_detector, motion_detector
    
    if not detectors_initialized:
        print("Initializing detectors...")
        face_detector = FaceDetector()
        age_detector = AgeDetector()
        emotion_detector = EmotionDetector()
        motion_detector = MotionDetector()
        detectors_initialized = True
        print("Detectors initialized.")


def process_frame(frame: np.ndarray, detect_motion: bool = True) -> dict:
    """
    Process a frame and return detection results.
    
    Args:
        frame: Input frame (BGR format)
        detect_motion: Whether to detect motion
        
    Returns:
        Detection results dictionary
    """
    results = {
        'faces': [],
        'motion': {}
    }
    
    # Detect faces
    faces = face_detector.detect(frame)
    
    # Process each face
    for face in faces:
        x, y, w, h = face['bbox']
        pad_h, pad_w = int(h * 0.2), int(w * 0.2)
        y1, y2 = max(0, y - pad_h), min(frame.shape[0], y + h + pad_h)
        x1, x2 = max(0, x - pad_w), min(frame.shape[1], x + w + pad_w)
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size > 0:
            # Detect age
            age_result = age_detector.detect(face_roi)
            
            # Detect emotion
            emotion_result = emotion_detector.detect(face_roi)
            
            # Combine results
            face_info = {
                'bbox': face['bbox'],
                'confidence': face['confidence'],
                'age': age_result,
                'emotion': emotion_result
            }
            
            results['faces'].append(face_info)
    
    # Detect motion
    if detect_motion:
        motion_result = motion_detector.detect(frame)
        results['motion'] = motion_result
    
    return results


def image_to_array(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to OpenCV array.
    
    Args:
        image_bytes: Image file bytes
        
    Returns:
        OpenCV image array (BGR format)
    """
    image = Image.open(io.BytesIO(image_bytes))
    # Convert RGB to BGR for OpenCV
    image_array = np.array(image)
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 3:  # RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    else:
        # Grayscale
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    
    return image_array


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Age, Emotion, and Motion Detection API",
        "version": config.API_VERSION,
        "endpoints": {
            "POST /detect/image": "Process a single image",
            "POST /detect/video": "Process a video file",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "detectors_initialized": detectors_initialized}


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    detect_motion: bool = False
):
    """
    Process a single image and return detection results.
    
    Args:
        file: Uploaded image file
        detect_motion: Whether to detect motion (default: False for single images)
        
    Returns:
        JSON response with detection results
    """
    try:
        # Initialize detectors if needed
        initialize_detectors()
        
        # Read image file
        image_bytes = await file.read()
        frame = image_to_array(image_bytes)
        
        # Process frame
        detections = process_frame(frame, detect_motion=detect_motion)
        
        # Format response
        result = APIOutput.format_detection_result(detections)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    detect_motion: bool = True,
    frame_skip: int = 1
):
    """
    Process a video file and return detection results.
    
    Args:
        file: Uploaded video file
        detect_motion: Whether to detect motion
        frame_skip: Process every Nth frame (1 = all frames)
        
    Returns:
        JSON response with detection results for all frames
    """
    try:
        # Initialize detectors if needed
        initialize_detectors()
        
        # Save uploaded video to temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Open video
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")
            
            results = []
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if configured
                if frame_number % frame_skip != 0:
                    frame_number += 1
                    continue
                
                # Process frame
                detections = process_frame(frame, detect_motion=detect_motion)
                result = APIOutput.format_detection_result(detections, frame_number)
                results.append(result)
                
                frame_number += 1
            
            cap.release()
            
            return JSONResponse(content={
                'success': True,
                'total_frames': len(results),
                'results': results
            })
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/detect/webcam")
async def detect_webcam(websocket: WebSocket):
    """
    WebSocket endpoint for real-time webcam detection.
    
    Note: This requires the client to send base64-encoded frames.
    """
    await websocket.accept()
    
    try:
        # Initialize detectors
        initialize_detectors()
        
        while True:
            # Receive frame data
            data = await websocket.receive_json()
            
            if 'frame' not in data:
                await websocket.send_json(APIOutput.format_error("No frame data provided"))
                continue
            
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(data['frame'])
                frame = image_to_array(image_bytes)
                
                # Process frame
                detect_motion = data.get('detect_motion', True)
                detections = process_frame(frame, detect_motion=detect_motion)
                
                # Format and send response
                result = APIOutput.format_detection_result(detections)
                await websocket.send_json(result)
            
            except Exception as e:
                await websocket.send_json(APIOutput.format_error(str(e)))
    
    except Exception as e:
        await websocket.send_json(APIOutput.format_error(str(e)))
    finally:
        await websocket.close()


def main():
    """Run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run detection API server")
    parser.add_argument(
        '--host',
        type=str,
        default=config.API_HOST,
        help=f'Host address (default: {config.API_HOST})'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=config.API_PORT,
        help=f'Port number (default: {config.API_PORT})'
    )
    
    args = parser.parse_args()
    
    print(f"Starting API server on {args.host}:{args.port}")
    print(f"API documentation available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
