"""
Data augmentation functions for paper corner detection using Albumentations.

This module provides robust augmentations that correctly transform both images
and keypoints (corners), crucial for training accurate corner detection models.
"""

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation(image_size=224, augmentation_strength='strong'):
    """
    Get Albumentations pipeline for training with corner-aware transforms.

    All geometric transforms automatically handle keypoint transformation,
    ensuring corners remain correctly aligned with augmented images.

    Args:
        image_size (int): Target image size (will resize to square). Default: 224
        augmentation_strength (str): Augmentation intensity:
            - 'light': Minimal augmentation (for large datasets)
            - 'medium': Moderate augmentation (default)
            - 'strong': Aggressive augmentation (for small datasets, recommended)

    Returns:
        albumentations.Compose: Augmentation pipeline

    Example:
        >>> aug = get_training_augmentation(image_size=224, augmentation_strength='strong')
        >>> # Corners should be in format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        >>> # with values in pixel coordinates
        >>> augmented = aug(image=image, keypoints=corners)
        >>> aug_image = augmented['image']  # Tensor [3, 224, 224]
        >>> aug_corners = augmented['keypoints']  # List of (x, y) tuples
    """

    if augmentation_strength == 'light':
        rotate_limit = 10
        perspective_scale = 0.05
        brightness_limit = 0.1
        contrast_limit = 0.1
        p_geometric = 0.3
    elif augmentation_strength == 'medium':
        rotate_limit = 15
        perspective_scale = 0.1
        brightness_limit = 0.15
        contrast_limit = 0.15
        p_geometric = 0.5
    else:  # strong (default for small datasets)
        rotate_limit = 25
        perspective_scale = 0.2
        brightness_limit = 0.2
        contrast_limit = 0.2
        p_geometric = 0.7

    transform = A.Compose([
        # Resize to target size (always applied)
        A.Resize(height=image_size, width=image_size),

        # Geometric transformations (with keypoint transformation)
        A.Rotate(
            limit=rotate_limit,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=p_geometric
        ),

        A.Perspective(
            scale=(0.02, perspective_scale),
            keep_size=True,
            p=p_geometric
        ),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.3),

        # ShiftScaleRotate for additional variation
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=rotate_limit,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=p_geometric
        ),

        # Photometric transformations (don't affect keypoints)
        A.ColorJitter(
            brightness=brightness_limit,
            contrast=contrast_limit,
            saturation=0.2,
            hue=0.1,
            p=0.8
        ),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        A.GaussNoise(std_range=(0.04, 0.2), p=0.3),  # Normalized std values [0,1]

        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.5
        ),

        # Normalize with ImageNet statistics
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        # Convert to PyTorch tensor
        ToTensorV2(),

    ], keypoint_params=A.KeypointParams(
        format='xy',  # Keypoints are (x, y) tuples
        remove_invisible=False,  # Keep keypoints even if outside image after aug
        label_fields=[]  # No labels, just coordinates
    ))

    return transform


def get_validation_augmentation(image_size=224):
    """
    Get Albumentations pipeline for validation/inference (no augmentation).

    Only applies resize and normalization, no random transforms.

    Args:
        image_size (int): Target image size. Default: 224

    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    transform = A.Compose([
        # Resize to target size
        A.Resize(height=image_size, width=image_size),

        # Normalize with ImageNet statistics
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        # Convert to PyTorch tensor
        ToTensorV2(),

    ], keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False,
        label_fields=[]
    ))

    return transform


def get_inference_transform(image_size=224):
    """
    Get transform for inference (alias for get_validation_augmentation).

    Args:
        image_size (int): Target image size. Default: 224

    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    return get_validation_augmentation(image_size)


def normalize_corners(corners, image_width, image_height):
    """
    Normalize corner coordinates to [0, 1] range.

    Args:
        corners (np.ndarray): Corner coordinates of shape (4, 2) in pixels
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels

    Returns:
        np.ndarray: Normalized corners of shape (4, 2) with values in [0, 1]
    """
    normalized = corners.copy().astype(np.float32)
    normalized[:, 0] /= image_width  # Normalize x
    normalized[:, 1] /= image_height  # Normalize y
    return normalized


def denormalize_corners(corners_normalized, image_width, image_height):
    """
    Denormalize corner coordinates from [0, 1] range to pixel coordinates.

    Args:
        corners_normalized (np.ndarray): Normalized corners of shape (4, 2) in [0, 1]
        image_width (int): Target image width in pixels
        image_height (int): Target image height in pixels

    Returns:
        np.ndarray: Corner coordinates of shape (4, 2) in pixels
    """
    denormalized = corners_normalized.copy().astype(np.float32)
    denormalized[:, 0] *= image_width  # Denormalize x
    denormalized[:, 1] *= image_height  # Denormalize y
    return denormalized
