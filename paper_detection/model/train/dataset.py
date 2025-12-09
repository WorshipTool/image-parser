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
from albumentations.pytorch import ToTensorV2

from paper_detection.model.train.config import TrainingConfig
from paper_detection.model.utils import IMAGE_MEAN, IMAGE_STD, preprocess_image


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

    def __init__(self, augment: bool = True):
        """
        Args:
            augment: Whether to apply augmentation
        """
        # Load config
        config = TrainingConfig()

        self.images_dir = Path(config.images_dir)
        self.image_size = config.image_size
        self.augment = augment

        # Load ground truth
        with open(config.corners_file, 'r') as f:
            self.ground_truth = json.load(f)

        self.image_names = list(self.ground_truth.keys())

        # Setup transforms
        self.setup_transforms()

    def setup_transforms(self):
        """Setup Albumentations transforms"""

        # Basic transform (always applied)
        basic_transform = [
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ]

        if self.augment:
            # Augmentation pipeline
            self.transform = A.Compose([
                # Geometric augmentations
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.GaussNoise(p=0.3),

                # Random rotation and flip
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),

                *basic_transform
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            # Validation transform (no augmentation)
            self.transform = A.Compose(
                basic_transform,
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
            )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """Get image and corners"""
        image_name = self.image_names[idx]

        # Load image
        image_path = self.images_dir / image_name
        image = cv2.imread(str(image_path))
        
        # Convert to RGB using preprocess_image
        image = preprocess_image(image)

        h, w = image.shape[:2]

        # Get ground truth corners (normalized 0-1)
        gt_data = self.ground_truth[image_name]
        corners_norm = np.array(gt_data["corners"], dtype=np.float32)

        # Ensure corners are in correct clockwise order (top-left, top-right, bottom-right, bottom-left)
        corners_norm = order_corners_clockwise(corners_norm)

        # Convert to pixel coordinates for Albumentations
        corners_px = corners_norm.copy()
        corners_px[:, 0] *= w
        corners_px[:, 1] *= h

        # Apply transforms
        transformed = self.transform(
            image=image,
            keypoints=corners_px
        )

        image_tensor = transformed['image']
        corners_transformed = np.array(transformed['keypoints'], dtype=np.float32)

        # Normalize corners back to 0-1 range (after resize to image_size)
        corners_normalized = corners_transformed / self.image_size

        # Flatten to [8] for model output
        corners_flat = corners_normalized.reshape(-1)

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
    train_size = int(config.train_split * dataset_size)
    val_size = dataset_size - train_size

    # Create separate datasets for train and val
    train_dataset = CornerDetectionDataset(
        augment=True  # Augmentation for training
    )

    val_dataset = CornerDetectionDataset(
        augment=False  # No augmentation for validation
    )

    # Use subset for train/val split
    indices = list(range(dataset_size))
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
