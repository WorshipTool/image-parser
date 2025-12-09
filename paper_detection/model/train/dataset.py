"""
Dataset for paper corner detection with Albumentations augmentation
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A

from paper_detection.model.train.config import TrainingConfig
from paper_detection.model.utils import preprocess_image, normalize_and_tensorize


def order_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    """
    Order corners in clockwise direction based on angle from centroid.
    
    The first corner will be the one closest to top-left corner of image (0, 0).
    This ensures consistent ordering regardless of paper rotation or perspective.
    
    Args:
        corners: Array of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    Returns:
        Ordered corners in clockwise direction starting from the corner closest to top-left
    """
    # Calculate centroid (center point)
    centroid = np.mean(corners, axis=0)
    
    # Calculate angle from centroid to each corner
    # Using atan2(y - cy, x - cx) where (cx, cy) is centroid
    # atan2 returns angles in range [-pi, pi]
    angles = np.arctan2(
        corners[:, 1] - centroid[1],  # dy
        corners[:, 0] - centroid[0]   # dx
    )
    
    # Sort by angle (clockwise order)
    sorted_indices = np.argsort(angles)
    sorted_corners = corners[sorted_indices]
    
    # Find which corner is closest to top-left (0, 0)
    distances_to_origin = np.sum(sorted_corners ** 2, axis=1)  # squared distance from (0,0)
    closest_idx = np.argmin(distances_to_origin)
    
    # Rotate array so closest corner is first
    ordered_corners = np.roll(sorted_corners, -closest_idx, axis=0)
    
    return ordered_corners


class CornerDetectionDataset(Dataset):
    """Dataset for corner detection with augmentation"""

    def __init__(self, augment: bool = True, max_images: int = None):
        """
        Args:
            augment: Whether to apply augmentation
            max_images: Maximum number of images to load (None = all images)
        """
        # Load config
        config = TrainingConfig()

        self.images_dir = Path(config.images_dir)
        self.image_size = config.image_size
        self.augment = augment

        # Load ground truth
        with open(config.corners_file, 'r') as f:
            self.ground_truth = json.load(f)

        # For debug, keep only first image
        self.image_names = list(self.ground_truth.keys())
        if max_images is not None:
            self.image_names = self.image_names[:max_images]

        # Setup transforms
        self.setup_transforms()

    def setup_transforms(self):
        """Setup Albumentations transforms"""

        # Basic transform (resize only - normalization/tensorization done separately)
        basic_transform = [
            A.Resize(self.image_size, self.image_size),
        ]

        if self.augment:
            # Augmentation pipeline
            self.transform = A.Compose([
                # Color/appearance augmentations
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.GaussNoise(p=0.3),

                # Geometric augmentations (these affect keypoints)
                A.ShiftScaleRotate(
                    shift_limit=0.05,   # ±5% shift
                    scale_limit=0.10,   # ±10% zoom
                    rotate_limit=90,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=0.9,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Perspective(scale=(0.03, 0.08), p=0.3),

                # Resize (must be after augmentations)
                *basic_transform
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            # Validation transform (no augmentation, only resize)
            self.transform = A.Compose(
                basic_transform,
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
            )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """Get image and corners"""
        image_name = self.image_names[idx]

        # Load image (BGR format from OpenCV)
        image_path = self.images_dir / image_name
        image_bgr = cv2.imread(str(image_path))

        # Convert BGR to RGB
        image_rgb = preprocess_image(image_bgr)

        h, w = image_rgb.shape[:2]

        # Get ground truth corners (normalized 0-1)
        # IMPORTANT: Keep ORIGINAL order from ground truth!
        # DO NOT reorder - this changes semantic meaning of each corner
        gt_data = self.ground_truth[image_name]
        corners_norm = np.array(gt_data["corners"], dtype=np.float32)

        # Ensure corners are in correct clockwise order
        corners_norm = order_corners_clockwise(corners_norm)

        # Convert to pixel coordinates for Albumentations
        corners_px = corners_norm.copy()
        corners_px[:, 0] *= w
        corners_px[:, 1] *= h

        # Apply Albumentations transforms (augmentations + resize)
        # Returns uint8 RGB image and transformed keypoints
        transformed = self.transform(
            image=image_rgb,
            keypoints=corners_px
        )

        image_augmented = transformed['image']  # RGB uint8, shape (224, 224, 3)
        corners_transformed = np.array(transformed['keypoints'], dtype=np.float32)

        # Normalize corners to 0-1 range (after resize to image_size)
        corners_normalized = corners_transformed / self.image_size

        # Flatten to [8] for model output
        corners_flat = corners_normalized.reshape(-1)

        # Apply unified normalization and tensorization
        # This ensures IDENTICAL preprocessing as inference
        image_tensor = normalize_and_tensorize(image_augmented)

        return {
            'image': image_tensor,
            'corners': torch.tensor(corners_flat, dtype=torch.float32),
            'image_name': image_name
        }


def create_dataloaders():
    """Create train and validation dataloaders"""

    # Load config
    config = TrainingConfig()

    # Load all data
    full_dataset = CornerDetectionDataset(
        augment=False  # Will set per split
    )

    # Split into train/val
    dataset_size = len(full_dataset)
    
    # Handle small datasets (for testing)
    if dataset_size < 2:
        print(f"⚠️  Warning: Dataset too small ({dataset_size} images). Using same image for train and val.")
        train_size = dataset_size
        val_size = dataset_size
    else:
        train_size = max(1, int(config.train_split * dataset_size))  # At least 1 for train
        val_size = max(1, dataset_size - train_size)  # At least 1 for val
        # Adjust train_size if val_size would be 0
        if val_size == 0:
            train_size = dataset_size - 1
            val_size = 1

    # Create separate datasets for train and val
    train_dataset = CornerDetectionDataset(
        augment=True  # Augmentation for training
    )

    val_dataset = CornerDetectionDataset(
        augment=False  # No augmentation for validation
    )

    # Use subset for train/val split
    indices = list(range(dataset_size))
    if dataset_size < 2:
        # Use same image for both train and val when dataset is too small
        train_indices = indices
        val_indices = indices
    else:
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader
