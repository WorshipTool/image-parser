"""
Corner-aware augmentation functions for paper detection training.

This module provides geometric augmentation functions that transform both
images and corner coordinates consistently. All geometric transforms work
on normalized coordinates in the range [0, 1].

Coordinate System:
    - Corners are represented as numpy arrays of shape [4, 2]
    - Each corner is [x, y] where x and y are in range [0, 1]
    - (0, 0) is top-left, (1, 1) is bottom-right
    - Corners order: [top-left, top-right, bottom-right, bottom-left]
"""

import numpy as np
import torch
import torchvision.transforms as T
import cv2
from typing import Tuple, Callable


def apply_horizontal_flip(image: np.ndarray, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flip image horizontally and transform corner coordinates accordingly.

    The transformation mirrors the image around the vertical axis, so x-coordinates
    are inverted while y-coordinates remain unchanged.

    Args:
        image: Input image as numpy array of shape [H, W, C]
        corners: Corner coordinates as numpy array of shape [4, 2] in normalized
                coordinates [0, 1]. Each corner is [x, y].

    Returns:
        Tuple of (flipped_image, transformed_corners)
        - flipped_image: Horizontally flipped image
        - transformed_corners: Corners with x-coordinates flipped (new_x = 1 - old_x)

    Example:
        >>> image = np.random.rand(224, 224, 3)
        >>> corners = np.array([[0.2, 0.1], [0.8, 0.1], [0.8, 0.9], [0.2, 0.9]])
        >>> flipped_img, flipped_corners = apply_horizontal_flip(image, corners)
        >>> print(flipped_corners)
        [[0.8, 0.1], [0.2, 0.1], [0.2, 0.9], [0.8, 0.9]]
    """
    # Flip image horizontally
    flipped_image = np.fliplr(image)

    # Transform corners: new_x = 1 - old_x (assuming normalized coords)
    flipped_corners = corners.copy()
    flipped_corners[:, 0] = 1.0 - corners[:, 0]

    return flipped_image, flipped_corners


def apply_vertical_flip(image: np.ndarray, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flip image vertically and transform corner coordinates accordingly.

    The transformation mirrors the image around the horizontal axis, so y-coordinates
    are inverted while x-coordinates remain unchanged.

    Args:
        image: Input image as numpy array of shape [H, W, C]
        corners: Corner coordinates as numpy array of shape [4, 2] in normalized
                coordinates [0, 1]. Each corner is [x, y].

    Returns:
        Tuple of (flipped_image, transformed_corners)
        - flipped_image: Vertically flipped image
        - transformed_corners: Corners with y-coordinates flipped (new_y = 1 - old_y)

    Example:
        >>> image = np.random.rand(224, 224, 3)
        >>> corners = np.array([[0.2, 0.1], [0.8, 0.1], [0.8, 0.9], [0.2, 0.9]])
        >>> flipped_img, flipped_corners = apply_vertical_flip(image, corners)
        >>> print(flipped_corners)
        [[0.2, 0.9], [0.8, 0.9], [0.8, 0.1], [0.2, 0.1]]
    """
    # Flip image vertically
    flipped_image = np.flipud(image)

    # Transform corners: new_y = 1 - old_y
    flipped_corners = corners.copy()
    flipped_corners[:, 1] = 1.0 - corners[:, 1]

    return flipped_image, flipped_corners


def apply_rotation(image: np.ndarray, corners: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image by angle degrees around center and transform corner coordinates.

    The rotation is performed around the image center (0.5, 0.5) in normalized
    coordinates. The transformation uses a 2D rotation matrix to update corner
    positions.

    Args:
        image: Input image as numpy array of shape [H, W, C]
        corners: Corner coordinates as numpy array of shape [4, 2] in normalized
                coordinates [0, 1]. Each corner is [x, y].
        angle: Rotation angle in degrees. Positive values rotate counter-clockwise.
              Typically in range [-15, 15] for training augmentation.

    Returns:
        Tuple of (rotated_image, transformed_corners)
        - rotated_image: Rotated image (same size as input, may have black borders)
        - transformed_corners: Corners rotated around center point (0.5, 0.5)

    Note:
        The rotation is performed around the center of the image. Corners may move
        slightly outside the [0, 1] range after rotation, which is acceptable for
        small rotation angles (±15°).

    Example:
        >>> image = np.random.rand(224, 224, 3)
        >>> corners = np.array([[0.2, 0.1], [0.8, 0.1], [0.8, 0.9], [0.2, 0.9]])
        >>> rotated_img, rotated_corners = apply_rotation(image, corners, 10.0)
    """
    h, w = image.shape[:2]

    # Get rotation matrix for image rotation (around image center in pixels)
    center_pixels = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center_pixels, angle, 1.0)

    # Rotate image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))

    # Transform corners using rotation matrix
    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Rotation matrix for normalized coordinates (around center point (0.5, 0.5))
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Center point in normalized coords
    center = np.array([0.5, 0.5])

    # Translate corners to origin, rotate, translate back
    rotated_corners = corners.copy()
    for i in range(len(corners)):
        # Translate to origin (center becomes 0,0)
        translated = corners[i] - center

        # Apply rotation matrix
        rotated = np.array([
            cos_a * translated[0] - sin_a * translated[1],
            sin_a * translated[0] + cos_a * translated[1]
        ])

        # Translate back
        rotated_corners[i] = rotated + center

    return rotated_image, rotated_corners


def get_training_augmentation(image_size: int = 224) -> Callable:
    """
    Returns a torchvision transform composition for training data augmentation.

    This transform applies color augmentation, blurring, normalization, and
    conversion to tensor. It does NOT include geometric augmentation (flip,
    rotation) as those need to be applied separately with corner transformation.

    The transforms are applied in the following order:
    1. Resize to target size
    2. ColorJitter for photometric augmentation
    3. GaussianBlur for robustness to focus variations
    4. Convert to tensor
    5. Normalize with ImageNet statistics

    Args:
        image_size: Target image size (height and width). Default: 224

    Returns:
        Callable transform that can be applied to PIL images or numpy arrays.
        The transform expects RGB images and returns normalized tensors.

    Note:
        For geometric augmentations (horizontal flip, vertical flip, rotation),
        use the dedicated functions (apply_horizontal_flip, apply_vertical_flip,
        apply_rotation) before applying this transform, as they also need to
        transform corner coordinates.

    Example:
        >>> transform = get_training_augmentation(224)
        >>> # Apply geometric augmentation first (with corners)
        >>> image_aug, corners_aug = apply_horizontal_flip(image, corners)
        >>> # Then apply photometric augmentation (corners unchanged)
        >>> image_tensor = transform(image_aug)
    """
    return T.Compose([
        T.ToPILImage(),  # Convert numpy array to PIL Image if needed
        T.Resize((image_size, image_size)),
        T.ColorJitter(
            brightness=0.2,  # Random brightness adjustment ±20%
            contrast=0.2,    # Random contrast adjustment ±20%
            saturation=0.2,  # Random saturation adjustment ±20%
            hue=0.1          # Random hue adjustment ±10%
        ),
        T.GaussianBlur(
            kernel_size=5,
            sigma=(0.1, 2.0)  # Random blur with sigma between 0.1 and 2.0
        ),
        T.ToTensor(),  # Convert to tensor and scale to [0, 1]
        T.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_inference_transform(image_size: int = 224) -> Callable:
    """
    Returns transform for inference (no augmentation).

    This transform only resizes, converts to tensor, and normalizes with
    ImageNet statistics. No augmentation is applied.

    Args:
        image_size: Target image size (height and width). Default: 224

    Returns:
        Callable transform that can be applied to PIL images or numpy arrays.
        The transform expects RGB images and returns normalized tensors.

    Example:
        >>> transform = get_inference_transform(224)
        >>> image_tensor = transform(image)
        >>> # image_tensor is ready for model inference
    """
    return T.Compose([
        T.ToPILImage(),  # Convert numpy array to PIL Image if needed
        T.Resize((image_size, image_size)),
        T.ToTensor(),  # Convert to tensor and scale to [0, 1]
        T.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics
            std=[0.229, 0.224, 0.225]
        )
    ])


# TODO: Add perspective transform augmentation (careful with corners going outside [0,1])
# Perspective transforms would simulate viewing the paper from different angles,
# which is very relevant for paper detection. However, care must be taken to:
# 1. Ensure corners don't go too far outside [0, 1] range
# 2. Keep the transform realistic (papers are typically viewed from moderate angles)
# 3. Consider clipping or rejecting transforms that create invalid corner positions
