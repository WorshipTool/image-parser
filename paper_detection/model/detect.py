"""
Corner detection inference using trained model
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Union, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

from paper_detection.model.train.model import CornerDetectionCNN
from paper_detection.model.train.config import TrainingConfig
from paper_detection.model.utils import preprocess_image, IMAGE_MEAN, IMAGE_STD
from paper_detection.model.config import IMAGE_SIZE


class CornerDetector:
    """Detector for paper corners using trained CNN model"""
    
    def __init__(self, model_path: Union[str, Path] = None, device: str = None):
        """
        Initialize corner detector
        
        Args:
            model_path: Path to trained model checkpoint. If None, uses default checkpoint.
            device: Device to run on ('cpu' or 'cuda'). If None, auto-detects.
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set model path
        if model_path is None:
            config = TrainingConfig()
            model_path = config.output_dir / "best_model.pth"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Load model
        self.model = CornerDetectionCNN(num_corners=4)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessing transform
        self.image_size = IMAGE_SIZE
        self.transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ])
        
        print(f"✓ Loaded model from: {model_path}")
        print(f"✓ Using device: {self.device}")
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect corners in image
        
        Args:
            image: Input image in BGR format (OpenCV format) or RGB
        
        Returns:
            Corners array of shape [4, 2] with pixel coordinates
            Order: clockwise starting from corner closest to top-left
        """
        # Store original size
        original_h, original_w = image.shape[:2]
        
        # Convert to RGB
        image_rgb = preprocess_image(image)
        
        # Apply resize, normalization and convert to tensor
        transformed = self.transform(image=image_rgb)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            corners_flat = self.model(image_tensor)  # [1, 8]
        
        # Reshape to [4, 2] and convert to numpy (normalized 0-1)
        corners_norm = corners_flat.cpu().numpy().reshape(4, 2)
        
        # Convert to pixel coordinates of original image
        corners_px = corners_norm.copy()
        corners_px[:, 0] *= original_w
        corners_px[:, 1] *= original_h
        
        return corners_px


def detect(image: Union[str, Path, np.ndarray], 
           model_path: Union[str, Path] = None) -> np.ndarray:
    """
    Convenience function to detect corners in an image
    
    Args:
        image: Path to image file or numpy array
        model_path: Path to model checkpoint. If None, uses default.
    
    Returns:
        Corners array of shape [4, 2] with pixel coordinates
        Order: clockwise starting from corner closest to top-left
    
    Example:
        >>> corners = detect("path/to/image.jpg")
        >>> print(corners.shape)  # (4, 2)
        >>> print(corners[0])  # [123.4, 567.8]
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise ValueError(f"Failed to load image: {image}")
    
    # Create detector (will be cached if called multiple times)
    detector = CornerDetector(model_path=model_path)
    
    # Detect corners
    return detector.detect(image)
