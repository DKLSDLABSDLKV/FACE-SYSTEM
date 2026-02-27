# Age, Emotion, Motion, and Movement Detection System

A comprehensive computer vision system that detects age, emotion, motion, and movement from webcam feeds, video files, and static images. The system provides real-time display, REST API endpoints, and file export capabilities.

## Features

- **Face Detection**: Accurate face detection using OpenCV DNN
- **Age Estimation**: Deep learning-based age estimation from facial features
- **Emotion Recognition**: 7-class emotion detection (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- **Motion Detection**: Background subtraction and optical flow for motion detection
- **Multiple Input Sources**: Webcam, video files, and static images
- **Real-time Display**: Live visualization with annotations
- **REST API**: FastAPI-based API for integration
- **File Export**: Save results as JSON or CSV

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for webcam mode)
- Internet connection (for automatic model downloads)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Webcam Mode

Process live webcam feed with real-time display:

```bash
python main.py --mode webcam --display
```

### Video File Mode

Process a video file and save results:

```bash
python main.py --mode video --input video.mp4 --output results.json
```

### Image Mode

Process a single image:

```bash
python main.py --mode image --input image.jpg --output results.json --display
```

## Usage

### Command Line Interface

```bash
python main.py [OPTIONS]
```

#### Options

- `--mode`: Input mode - `webcam`, `video`, or `image` (required)
- `--input`: Input path (required for video/image mode)
- `--output`: Output path for saving results (JSON or CSV)
- `--display`: Enable real-time display
- `--camera`: Camera index for webcam mode (default: 0)

#### Examples

```bash
# Webcam with display and save results
python main.py --mode webcam --display --output webcam_results.json

# Process video file
python main.py --mode video --input path/to/video.mp4 --output video_results.csv

# Process image with display
python main.py --mode image --input path/to/image.jpg --display

# Use different camera
python main.py --mode webcam --camera 1 --display
```

### Keyboard Controls (Display Mode)

- `q`: Quit application
- `s`: Save screenshot

## REST API

### Start API Server

```bash
python api_server.py --port 8000
```

The API documentation will be available at `http://localhost:8000/docs`

### API Endpoints

#### `GET /`

Root endpoint with API information.

#### `GET /health`

Health check endpoint.

#### `POST /detect/image`

Process a single image file.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file), `detect_motion` (optional boolean)

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00",
  "faces": [
    {
      "bbox": [x, y, w, h],
      "confidence": 0.95,
      "age": {
        "age": 25,
        "age_range": [15, 20],
        "confidence": 0.85
      },
      "emotion": {
        "emotion": "Happy",
        "confidence": 0.92,
        "probabilities": {...}
      }
    }
  ],
  "motion": {
    "detected": false,
    "magnitude": 0.0
  }
}
```

#### `POST /detect/video`

Process a video file.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (video file), `detect_motion` (optional boolean), `frame_skip` (optional int)

**Response:**
```json
{
  "success": true,
  "total_frames": 100,
  "results": [...]
}
```

#### `WebSocket /detect/webcam`

Real-time webcam detection via WebSocket.

**Connection:** `ws://localhost:8000/detect/webcam`

**Send:**
```json
{
  "frame": "base64_encoded_image",
  "detect_motion": true
}
```

**Receive:**
```json
{
  "success": true,
  "faces": [...],
  "motion": {...}
}
```

### Example API Usage

#### Using curl

```bash
# Detect in image
curl -X POST "http://localhost:8000/detect/image" \
  -F "file=@image.jpg" \
  -F "detect_motion=false"

# Detect in video
curl -X POST "http://localhost:8000/detect/video" \
  -F "file=@video.mp4" \
  -F "detect_motion=true" \
  -F "frame_skip=1"
```

#### Using Python

```python
import requests

# Process image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect/image',
        files={'file': f},
        data={'detect_motion': False}
    )
    print(response.json())
```

## Project Structure

```
FACE/
├── detectors/
│   ├── __init__.py
│   ├── face_detector.py      # Face detection module
│   ├── age_detector.py        # Age estimation module
│   ├── emotion_detector.py    # Emotion recognition module
│   └── motion_detector.py     # Motion detection module
├── input_handler.py           # Unified input handling
├── output_handler.py          # Display, API, file export
├── main.py                    # Main application
├── api_server.py              # FastAPI REST API server
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── models/                    # Pre-trained model files (auto-downloaded)
├── README.md                  # This file
└── examples/                  # Example usage scripts
    ├── webcam_example.py
    ├── video_example.py
    └── image_example.py
```

## Configuration

Edit `config.py` to customize:

- Model paths and URLs
- Detection thresholds
- Display settings
- API settings
- Frame processing parameters

## Models

The system automatically downloads required models on first run:

- **Face Detection**: OpenCV's face_detection_yunet.onnx
- **Age Detection**: Age estimation model
- **Emotion Detection**: Facial expression recognition model

Models are stored in the `models/` directory.

## Output Formats

### JSON Format

```json
{
  "metadata": {
    "total_frames": 100,
    "exported_at": "2024-01-01T12:00:00"
  },
  "results": [
    {
      "frame_number": 0,
      "timestamp": "2024-01-01T12:00:00",
      "detections": {
        "faces": [...],
        "motion": {...}
      }
    }
  ]
}
```

### CSV Format

| frame_number | timestamp | num_faces | motion_detected | motion_magnitude |
|--------------|-----------|-----------|-----------------|------------------|
| 0 | 2024-01-01T12:00:00 | 1 | false | 0.0 |

## Troubleshooting

### Models Not Downloading

If models fail to download automatically:

1. Check your internet connection
2. Manually download models from the URLs in `config.py`
3. Place them in the `models/` directory

### Webcam Not Working

- Check if webcam is connected
- Try different camera indices: `--camera 1`, `--camera 2`, etc.
- On Linux, ensure proper permissions for `/dev/video*`

### Performance Issues

- Reduce frame processing by setting `FRAME_SKIP` in `config.py`
- Reduce maximum frame size in `config.py`
- Use GPU acceleration if available (modify detector initialization)

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- OpenCV for computer vision libraries
- FastAPI for the REST API framework
- Pre-trained models from OpenCV Zoo
