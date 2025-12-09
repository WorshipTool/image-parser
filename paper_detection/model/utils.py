"""
Utility functions for image preprocessing
"""

import cv2
import numpy as np
from typing import Tuple

# ImageNet normalization values (used for pretrained models)
IMAGE_MEAN = [0.485, 0.456, 0.406]  # RGB mean
IMAGE_STD = [0.229, 0.224, 0.225]   # RGB std


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image - convert to RGB format
    
    Ensures image is in RGB format (3 channels) regardless of input format.
    
    Args:
        image: Input image (can be BGR, RGB, grayscale, or RGBA)
    
    Returns:
        RGB image with 3 channels
    """
    # Convert to RGB if needed
    if len(image.shape) == 2:
        # Grayscale to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        # Assume BGR (OpenCV default) and convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[2] == 4:
        # RGBA to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
    
    return image_rgb

