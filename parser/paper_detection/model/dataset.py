"""
Dataset for paper segmentation with mask generation from corners
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from paper_detection.model.config import ModelConfig


def generate_mask_from_corners(corners: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    Generate binary mask from 4 corner points by filling polygon

    Args:
        corners: Array of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                 in pixel coordinates
        image_shape: Shape of the image (height, width)

    Returns:
        Binary mask [H, W] with 255 inside polygon, 0 outside
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Convert corners to integer pixel coordinates
    corners_int = corners.astype(np.int32)

    # Fill polygon
    cv2.fillPoly(mask, [corners_int], 255)

    return mask


class SegmentationDataset(Dataset):
    """
    Dataset for paper segmentation

    Loads images and generates ground truth masks from corner annotations
    """

    def __init__(self, config: ModelConfig, augment: bool = True, max_images: int = None):
        """
        Args:
            config: Segmentation configuration
            augment: Whether to apply augmentation
            max_images: Maximum number of images to load (None = all images)
        """
        self.config = config
        self.images_dir = Path(config.IMAGES_DIR)
        self.image_size = config.IMAGE_SIZE
        self.augment = augment

        # Load ground truth corners
        with open(config.CORNERS_FILE, 'r') as f:
            self.ground_truth = json.load(f)

        # Load allowed image names from dataset.json if it exists
        dataset_json_path = Path(config.DATASET_FILE)
        if dataset_json_path.exists():
            with open(dataset_json_path, 'r') as f:
                allowed_names = set(json.load(f))
            # Only keep image names present in both corners.json and dataset.json
            self.image_names = [name for name in self.ground_truth.keys() if name in allowed_names]
        else:
            self.image_names = list(self.ground_truth.keys())

        if max_images is not None:
            self.image_names = self.image_names[:max_images]

        # Setup transforms
        self.setup_transforms()

    def setup_transforms(self):
        """Setup Albumentations transforms"""

        # Basic transform (resize only)
        basic_transform = [
            A.Resize(self.image_size, self.image_size),
        ]

        if self.augment:
            # Augmentation pipeline
            self.transform = A.Compose([
                # Color/appearance augmentations (don't affect mask or keypoints)
                A.RandomBrightnessContrast(p=self.config.AUG_BRIGHTNESS_CONTRAST_P),
                A.HueSaturationValue(p=self.config.AUG_HUE_SAT_P),
                A.GaussianBlur(blur_limit=(3, 7), p=self.config.AUG_BLUR_P),
                A.GaussNoise(p=self.config.AUG_NOISE_P),

                # Geometric augmentations (affect image, mask, AND keypoints)
                A.ShiftScaleRotate(
                    shift_limit=0.05,   # ±5% shift
                    scale_limit=0.10,   # ±10% zoom
                    rotate_limit=90,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=self.config.AUG_ROTATE_P,
                ),
                A.HorizontalFlip(p=self.config.AUG_HFLIP_P),
                A.VerticalFlip(p=self.config.AUG_VFLIP_P),
                A.Perspective(
                    scale=(0.03, 0.08),
                    p=self.config.AUG_PERSPECTIVE_P,
                    fit_output=False
                ),

                # Resize (must be after augmentations)
                *basic_transform,

                # Normalize and convert to tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            # Validation transform (no augmentation, only resize and normalize)
            self.transform = A.Compose([
                *basic_transform,
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """Get image and mask"""
        image_name = self.image_names[idx]

        # Load image (BGR format from OpenCV)
        image_path = self.images_dir / image_name
        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        h, w = image_rgb.shape[:2]

        # Get ground truth corners (normalized 0-1)
        gt_data = self.ground_truth[image_name]
        corners_norm = np.array(gt_data["corners"], dtype=np.float32)

        # Convert to pixel coordinates
        corners_px = corners_norm.copy()
        corners_px[:, 0] *= w
        corners_px[:, 1] *= h

        # Generate ground truth mask from corners
        mask = generate_mask_from_corners(corners_px, image_rgb.shape)

        # Apply Albumentations transforms (augmentations + resize + normalize)
        # IMPORTANT: Albumentations handles image, mask, AND keypoints together
        transformed = self.transform(image=image_rgb, mask=mask, keypoints=corners_px)

        image_tensor = transformed['image']  # [3, H, W], normalized
        mask_tensor = transformed['mask']    # [H, W], uint8 (0 or 255)
        corners_transformed = np.array(transformed['keypoints'], dtype=np.float32)  # [4, 2]

        # Convert mask to float [0, 1] and add channel dimension
        mask_float = mask_tensor.float() / 255.0  # [H, W] -> [0, 1]
        mask_float = mask_float.unsqueeze(0)      # [H, W] -> [1, H, W]

        # Also return ground truth corners (at model resolution) for evaluation
        # corners_transformed are already at IMAGE_SIZE resolution from Albumentations
        corners_tensor = torch.tensor(corners_transformed, dtype=torch.float32)

        return {
            'image': image_tensor,           # [3, H, W]
            'mask': mask_float,              # [1, H, W]
            'corners': corners_tensor,       # [4, 2] in pixel coordinates at IMAGE_SIZE
            'image_name': image_name
        }


def create_dataloaders(config: ModelConfig):
    """
    Create train and validation dataloaders

    Args:
        config: Segmentation configuration

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load all data
    full_dataset = SegmentationDataset(config, augment=False)

    # Split into train/val
    dataset_size = len(full_dataset)

    # Handle small datasets (for testing)
    if dataset_size < 2:
        print(f"⚠️  Warning: Dataset too small ({dataset_size} images). Using same image for train and val.")
        train_size = dataset_size
        val_size = dataset_size
    else:
        train_size = max(1, int(config.TRAIN_SPLIT * dataset_size))  # At least 1 for train
        val_size = max(1, dataset_size - train_size)  # At least 1 for val
        # Adjust train_size if val_size would be 0
        if val_size == 0:
            train_size = dataset_size - 1
            val_size = 1

    # Create separate datasets for train and val
    train_dataset = SegmentationDataset(config, augment=True)   # Augmentation for training
    val_dataset = SegmentationDataset(config, augment=False)    # No augmentation for validation

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
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Dataset split: {train_size} train, {val_size} val (total: {dataset_size})")

    return train_loader, val_loader
