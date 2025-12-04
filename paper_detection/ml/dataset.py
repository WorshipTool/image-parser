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
    Training mode applies geometric augmentations randomly:
    1. Horizontal flip (50% probability) - mirrors image and corners horizontally
    2. Vertical flip (50% probability) - mirrors image and corners vertically
    3. Rotation ±15° (50% probability) - rotates image and corners around center
    4. Color augmentations (via transform parameter) - does not affect corners
    5. Normalization (via transform parameter) - prepares for model input

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
from PIL import Image
from typing import Dict, List, Optional, Callable

from .config import TrainingConfig
from .augmentations import (
    apply_horizontal_flip,
    apply_vertical_flip,
    apply_rotation,
    get_training_augmentation,
    get_inference_transform
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
        transform: Optional callable for photometric augmentations and normalization.
                   If None, appropriate default transform will be used based on mode.

    Attributes:
        images_dir: Directory containing images
        ground_truth_path: Path to ground truth JSON
        config: Training configuration
        mode: Current mode ("train" or "val")
        transform: Transform pipeline to apply
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
        transform: Optional[Callable] = None
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

        # Set transform (use defaults if not provided)
        if transform is None:
            if mode == "train":
                self.transform = get_training_augmentation(config.image_size)
            else:
                self.transform = get_inference_transform(config.image_size)
        else:
            self.transform = transform

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
        4. Applies geometric augmentations if in training mode
        5. Resizes image to target size
        6. Applies photometric augmentations and normalization
        7. Returns image tensor and flattened corner coordinates

        Args:
            idx: Index of sample to retrieve

        Returns:
            Dictionary containing:
                - "image": Normalized image tensor of shape [3, H, W]
                - "corners": Flattened corner coordinates of shape [8]
                - "image_name": Filename of the image

        Note:
            Geometric augmentations are applied with 50% probability each during training.
            Multiple augmentations can be applied to the same sample (e.g., both flip and rotation).
        """
        # Get image name
        image_name = self.image_names[idx]

        # Load image using OpenCV (returns BGR format)
        image_path = os.path.join(self.images_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get corners from ground truth (already normalized [0,1])
        corners = np.array(self.ground_truth[image_name]["corners"], dtype=np.float32)

        # Apply geometric augmentations randomly if in training mode
        if self.mode == "train":
            # Horizontal flip (50% probability)
            if np.random.random() < 0.5:
                image, corners = apply_horizontal_flip(image, corners)

            # Vertical flip (50% probability)
            if np.random.random() < 0.5:
                image, corners = apply_vertical_flip(image, corners)

            # Rotation ±15° (50% probability)
            if np.random.random() < 0.5:
                angle = np.random.uniform(-15, 15)
                image, corners = apply_rotation(image, corners, angle)

        # Resize image to target size
        image = cv2.resize(image, (self.config.image_size, self.config.image_size))

        # Apply transform (color augmentations + normalization)
        if self.transform:
            image = self.transform(image)

        # Convert corners to torch.FloatTensor and flatten [4, 2] -> [8]
        corners_flat = torch.FloatTensor(corners.flatten())

        return {
            "image": image,
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


# TODO: Add more sophisticated augmentations when dataset grows (perspective, crop, scale)
# Currently we use basic augmentations (flip, rotation) that are sufficient for small datasets.
# As the dataset grows, consider adding:
# - Perspective transforms to simulate different viewing angles
# - Random cropping with corner adjustment
# - Scale/zoom variations
# - Elastic deformations
# - More aggressive color augmentations
# These would help the model generalize better to diverse real-world conditions.

# TODO: Consider K-fold cross-validation for better evaluation
# With a small dataset, K-fold cross-validation would provide more robust
# performance estimates and better utilize all available data. This would involve:
# 1. Splitting data into K folds
# 2. Training K models, each using a different fold for validation
# 3. Averaging performance metrics across all folds
# 4. Selecting the best model based on average validation performance
# This approach is especially valuable when dataset size is limited (< 100 samples).
