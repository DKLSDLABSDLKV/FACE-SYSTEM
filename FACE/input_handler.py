"""
Unified input handler for webcam, video files, and images.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Iterator, Tuple
from abc import ABC, abstractmethod
import config


class InputHandler(ABC):
    """Abstract base class for input handlers."""
    
    @abstractmethod
    def get_frame(self) -> Optional[Tuple[np.ndarray, bool]]:
        """
        Get next frame from input source.
        
        Returns:
            Tuple of (frame, has_more) where:
            - frame: Image frame (BGR format) or None if no frame available
            - has_more: Boolean indicating if more frames are available
        """
        pass
    
    @abstractmethod
    def release(self):
        """Release input resources."""
        pass
    
    @abstractmethod
    def get_fps(self) -> float:
        """Get frames per second of the input source."""
        pass
    
    @abstractmethod
    def get_size(self) -> Tuple[int, int]:
        """Get frame size (width, height)."""
        pass


class WebcamInput(InputHandler):
    """Webcam input handler."""
    
    def __init__(self, camera_index: int = None, width: int = None, height: int = None):
        """
        Initialize webcam input.
        
        Args:
            camera_index: Camera index (default from config)
            width: Desired frame width
            height: Desired frame height
        """
        self.camera_index = camera_index if camera_index is not None else config.WEBCAM_INDEX
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        # Set resolution if specified
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Get actual resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        if self.fps <= 0:
            self.fps = config.WEBCAM_FPS
        
        print(f"Webcam initialized: {self.width}x{self.height} @ {self.fps} FPS")
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, bool]]:
        """Get next frame from webcam."""
        ret, frame = self.cap.read()
        if ret:
            return (frame, True)
        return (None, False)
    
    def release(self):
        """Release webcam."""
        if self.cap is not None:
            self.cap.release()
    
    def get_fps(self) -> float:
        """Get webcam FPS."""
        return self.fps
    
    def get_size(self) -> Tuple[int, int]:
        """Get frame size."""
        return (self.width, self.height)


class VideoInput(InputHandler):
    """Video file input handler."""
    
    def __init__(self, video_path: str):
        """
        Initialize video input.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
        print(f"Video loaded: {self.width}x{self.height} @ {self.fps} FPS, {self.frame_count} frames")
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, bool]]:
        """Get next frame from video."""
        ret, frame = self.cap.read()
        self.current_frame += 1
        
        if ret:
            has_more = self.current_frame < self.frame_count
            return (frame, has_more)
        
        return (None, False)
    
    def release(self):
        """Release video file."""
        if self.cap is not None:
            self.cap.release()
    
    def get_fps(self) -> float:
        """Get video FPS."""
        return self.fps
    
    def get_size(self) -> Tuple[int, int]:
        """Get frame size."""
        return (self.width, self.height)
    
    def get_frame_count(self) -> int:
        """Get total number of frames."""
        return self.frame_count
    
    def get_current_frame(self) -> int:
        """Get current frame number."""
        return self.current_frame
    
    def reset(self):
        """Reset video to beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0


class ImageInput(InputHandler):
    """Image file input handler."""
    
    def __init__(self, image_path: str):
        """
        Initialize image input.
        
        Args:
            image_path: Path to image file
        """
        self.image_path = Path(image_path)
        
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        self.frame = cv2.imread(str(self.image_path))
        
        if self.frame is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        
        self.height, self.width = self.frame.shape[:2]
        self.fps = 1.0  # Single image, so FPS is not applicable
        self.frame_read = False
        
        print(f"Image loaded: {self.width}x{self.height}")
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, bool]]:
        """Get image frame."""
        if not self.frame_read:
            self.frame_read = True
            return (self.frame.copy(), False)
        return (None, False)
    
    def release(self):
        """Release image (no-op for images)."""
        pass
    
    def get_fps(self) -> float:
        """Get FPS (not applicable for images)."""
        return self.fps
    
    def get_size(self) -> Tuple[int, int]:
        """Get image size."""
        return (self.width, self.height)


def create_input_handler(input_type: str, input_path: Optional[str] = None, **kwargs) -> InputHandler:
    """
    Factory function to create appropriate input handler.
    
    Args:
        input_type: Type of input - "webcam", "video", or "image"
        input_path: Path to video/image file (required for video/image types)
        **kwargs: Additional arguments for input handler
        
    Returns:
        InputHandler instance
    """
    if input_type.lower() == "webcam":
        return WebcamInput(**kwargs)
    elif input_type.lower() == "video":
        if input_path is None:
            raise ValueError("input_path is required for video input")
        return VideoInput(input_path)
    elif input_type.lower() == "image":
        if input_path is None:
            raise ValueError("input_path is required for image input")
        return ImageInput(input_path)
    else:
        raise ValueError(f"Unknown input type: {input_type}. Must be 'webcam', 'video', or 'image'")
