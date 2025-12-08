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


class CornerDetectionDataset(Dataset):
    """Dataset for corner detection with augmentation"""

    def __init__(
        self,
        images_dir: Path,
        ground_truth_file: Path,
        image_size: int = 224,
        augment: bool = True
    ):
        """
        Args:
            images_dir: Directory containing images
            ground_truth_file: JSON file with ground truth corners
            image_size: Size to resize images to
            augment: Whether to apply augmentation
        """
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.augment = augment

        # Load ground truth
        with open(ground_truth_file, 'r') as f:
            self.ground_truth = json.load(f)

        self.image_names = list(self.ground_truth.keys())

        # Setup transforms
        self.setup_transforms()

    def setup_transforms(self):
        """Setup Albumentations transforms"""

        # Basic transform (always applied)
        basic_transform = [
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # Get ground truth corners (normalized 0-1)
        gt_data = self.ground_truth[image_name]
        corners_norm = np.array(gt_data["corners"], dtype=np.float32)

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


def create_dataloaders(config, train_split=0.8):
    """Create train and validation dataloaders"""

    # Load all data
    full_dataset = CornerDetectionDataset(
        images_dir=config.images_dir,
        ground_truth_file=config.ground_truth_file,
        image_size=config.image_size,
        augment=False  # Will set per split
    )

    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    # Create separate datasets for train and val
    train_dataset = CornerDetectionDataset(
        images_dir=config.images_dir,
        ground_truth_file=config.ground_truth_file,
        image_size=config.image_size,
        augment=True  # Augmentation for training
    )

    val_dataset = CornerDetectionDataset(
        images_dir=config.images_dir,
        ground_truth_file=config.ground_truth_file,
        image_size=config.image_size,
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
