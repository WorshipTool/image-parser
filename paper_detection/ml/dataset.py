"""
PyTorch dataset for paper corner detection training.

This module provides a Dataset class that loads images and ground truth corner
annotations for training and validation of corner detection models.

Ground Truth Format:
    The ground truth JSON file contains a dictionary mapping image filenames to
    corner annotations:
    {
        "image_name.jpg": {
            "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            "tolerance_pixels": 50,
            "note": "Optional description"
        },
        ...
    }

Corner Coordinate System:
    - Corners are stored as normalized coordinates in range [0, 1]
    - (0, 0) represents top-left corner of image
    - (1, 1) represents bottom-right corner of image
    - Each corner is [x, y] where x is horizontal and y is vertical
    - Corners can slightly exceed [0, 1] range if paper extends beyond image edges

Augmentation Pipeline:
    Uses Albumentations library for robust geometric and photometric augmentations.
    Training mode applies:
    - Geometric transforms: rotation, perspective, flips, shift/scale/rotate
    - Photometric transforms: color jitter, blur, noise, brightness/contrast
    - All geometric transforms automatically handle keypoint (corner) transformation
    - Normalization with ImageNet statistics

    Validation mode applies only:
    - Resize to target size
    - Normalization with ImageNet statistics

Output Format:
    Each __getitem__ call returns a dictionary:
    {
        "image": torch.FloatTensor of shape [3, H, W] (normalized, RGB),
        "corners": torch.FloatTensor of shape [8] (flattened x1,y1,x2,y2,x3,y3,x4,y4),
        "image_name": str (filename without path)
    }
"""

import json
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional

from .config import TrainingConfig
from .augmentations import (
    get_training_augmentation,
    get_validation_augmentation
)


class PaperCornersDataset(Dataset):
    """
    PyTorch Dataset for paper corner detection.

    This dataset loads images and their corresponding corner annotations from
    a JSON ground truth file. It supports train/val splitting, geometric
    augmentations, and photometric augmentations.

    Args:
        images_dir: Path to directory containing test images
        ground_truth_path: Path to JSON file with corner annotations
        config: TrainingConfig object with dataset parameters
        mode: Either "train" or "val" for training/validation split
        augmentation_strength: Augmentation intensity for training mode.
                              Options: 'light', 'medium', 'strong' (default: 'strong')
                              Only used when mode='train'

    Attributes:
        images_dir: Directory containing images
        ground_truth_path: Path to ground truth JSON
        config: Training configuration
        mode: Current mode ("train" or "val")
        transform: Albumentations transform pipeline with keypoint support
        ground_truth: Loaded ground truth dictionary
        image_names: List of image filenames for current mode (train or val)

    Example:
        >>> config = TrainingConfig(image_size=224, train_split=0.8, seed=42)
        >>> train_dataset = PaperCornersDataset(
        ...     images_dir="paper_detection/tests/test_images",
        ...     ground_truth_path="paper_detection/tests/test_corners_ground_truth.json",
        ...     config=config,
        ...     mode="train"
        ... )
        >>> print(f"Training samples: {len(train_dataset)}")
        >>> sample = train_dataset[0]
        >>> print(f"Image shape: {sample['image'].shape}")
        >>> print(f"Corners shape: {sample['corners'].shape}")
    """

    def __init__(
        self,
        images_dir: str,
        ground_truth_path: str,
        config: TrainingConfig,
        mode: str = "train",
        augmentation_strength: str = "strong"
    ):
        """Initialize dataset with images, ground truth, and configuration."""
        assert mode in ["train", "val"], f"Mode must be 'train' or 'val', got '{mode}'"

        self.images_dir = images_dir
        self.ground_truth_path = ground_truth_path
        self.config = config
        self.mode = mode

        # Load ground truth JSON
        with open(ground_truth_path, 'r') as f:
            self.ground_truth: Dict = json.load(f)

        # Get all image names from ground truth
        all_image_names = list(self.ground_truth.keys())

        # Create deterministic train/val split
        np.random.seed(config.seed)
        shuffled_indices = np.random.permutation(len(all_image_names))

        # Split into train and validation sets
        train_size = int(len(all_image_names) * config.train_split)
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]

        # Store only images for current mode
        if mode == "train":
            self.image_names = [all_image_names[i] for i in train_indices]
        else:  # val
            self.image_names = [all_image_names[i] for i in val_indices]

        # Create Albumentations transform based on mode
        if mode == "train":
            self.transform = get_training_augmentation(
                config.image_size,
                augmentation_strength=augmentation_strength
            )
        else:  # val
            self.transform = get_validation_augmentation(config.image_size)

        print(f"Initialized {mode} dataset with {len(self.image_names)} images")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a single sample from the dataset.

        This method:
        1. Loads the image from disk (BGR format)
        2. Converts BGR to RGB
        3. Gets ground truth corners (normalized [0,1])
        4. Denormalizes corners to pixel coordinates
        5. Applies Albumentations transform (resize, augmentation, normalization)
           - All geometric transforms automatically handle keypoint transformation
        6. Normalizes augmented corners back to [0,1]
        7. Returns image tensor and flattened corner coordinates

        Args:
            idx: Index of sample to retrieve

        Returns:
            Dictionary containing:
                - "image": Normalized image tensor of shape [3, H, W]
                - "corners": Flattened corner coordinates of shape [8]
                - "image_name": Filename of the image

        Note:
            Albumentations handles all augmentations (geometric and photometric)
            and automatically transforms keypoints (corners) correctly.
        """
        # Get image name
        image_name = self.image_names[idx]

        # Load image using OpenCV (returns BGR format)
        image_path = os.path.join(self.images_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get original image dimensions
        image_height, image_width = image.shape[:2]

        # Get corners from ground truth (already normalized [0,1])
        corners_normalized = np.array(
            self.ground_truth[image_name]["corners"],
            dtype=np.float32
        )

        # Denormalize corners to pixel coordinates for Albumentations
        corners_px = corners_normalized.copy()
        corners_px[:, 0] *= image_width  # Denormalize x
        corners_px[:, 1] *= image_height  # Denormalize y

        # Convert corners to list of tuples for Albumentations keypoints format
        keypoints = [(x, y) for x, y in corners_px]

        # Apply Albumentations transform (handles resize, augmentation, normalization)
        augmented = self.transform(image=image_rgb, keypoints=keypoints)

        # Extract augmented image and keypoints
        image_tensor = augmented['image']  # Already a torch.Tensor [3, H, W]
        aug_keypoints = augmented['keypoints']  # List of (x, y) tuples

        # Convert keypoints back to numpy array and normalize to [0,1]
        corners_aug = np.array(aug_keypoints, dtype=np.float32)  # Shape: (4, 2)
        corners_norm = corners_aug / self.config.image_size  # Normalize to [0, 1]

        # Flatten corners from (4, 2) to (8,)
        corners_flat = torch.FloatTensor(corners_norm.flatten())

        return {
            "image": image_tensor,
            "corners": corners_flat,
            "image_name": image_name
        }

    @property
    def num_train_images(self) -> int:
        """
        Return the total number of training images in the dataset.

        This property calculates the number of images that would be in the
        training set based on the train_split ratio, regardless of the
        current mode of this dataset instance.

        Returns:
            Number of training images (int)
        """
        total_images = len(self.ground_truth)
        return int(total_images * self.config.train_split)

    @property
    def num_val_images(self) -> int:
        """
        Return the total number of validation images in the dataset.

        This property calculates the number of images that would be in the
        validation set based on the train_split ratio, regardless of the
        current mode of this dataset instance.

        Returns:
            Number of validation images (int)
        """
        total_images = len(self.ground_truth)
        train_images = int(total_images * self.config.train_split)
        return total_images - train_images
