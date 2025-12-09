"""
Utility functions for image preprocessing
"""

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

# Type definitions
Corners = NDArray[np.float32]  # Shape: [4, 2] - 4 corners with (x, y) coordinates

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


def normalize_and_tensorize(image_rgb: np.ndarray) -> torch.Tensor:
    """
    Normalize and convert RGB image to tensor (HWC → CHW).

    This is a helper function used after augmentations/resize to ensure
    consistent normalization between training and inference.

    Args:
        image_rgb: RGB image (already resized) - shape (H, W, 3), values 0-255

    Returns:
        Normalized tensor - shape (3, H, W)
    """
    # Convert to float32 and normalize to [0, 1]
    image_float = image_rgb.astype(np.float32) / 255.0

    # Normalize with ImageNet mean/std
    mean = np.array(IMAGE_MEAN, dtype=np.float32)
    std = np.array(IMAGE_STD, dtype=np.float32)
    image_normalized = (image_float - mean) / std

    # Convert HWC → CHW (Height, Width, Channels → Channels, Height, Width)
    image_chw = np.transpose(image_normalized, (2, 0, 1))

    # Convert to PyTorch tensor
    image_tensor = torch.from_numpy(image_chw).float()

    return image_tensor


def preprocess_image_for_model(image_bgr: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """
    Unified preprocessing function for model input.

    This function ensures IDENTICAL preprocessing for both training and inference:
    1. BGR → RGB conversion
    2. Resize to model input size (224×224)
    3. Normalize with ImageNet mean/std
    4. Convert HWC → CHW format
    5. Convert to PyTorch tensor

    Args:
        image_bgr: Input image in BGR format (OpenCV default) - shape (H, W, 3)
        image_size: Target size for model input (default: 224)

    Returns:
        Preprocessed image tensor ready for model input - shape (3, image_size, image_size)

    Example:
        >>> image = cv2.imread("image.jpg")  # BGR format
        >>> tensor = preprocess_image_for_model(image)
        >>> tensor.shape
        torch.Size([3, 224, 224])
    """
    # Step 1: Convert BGR → RGB
    image_rgb = preprocess_image(image_bgr)

    # Step 2: Resize to model input size
    image_resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # Step 3-5: Normalize and convert to tensor (using shared function)
    image_tensor = normalize_and_tensorize(image_resized)

    return image_tensor

