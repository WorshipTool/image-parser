"""
Paper Segmentation Module

U-Net based segmentation for paper detection with corner extraction
"""

from paper_detection.segmentation.config import SegmentationConfig
from paper_detection.segmentation.model import UNet, create_unet, CombinedLoss, DiceLoss
from paper_detection.segmentation.dataset import SegmentationDataset, create_dataloaders, generate_mask_from_corners
from paper_detection.segmentation.postprocess import (
    mask_to_corners,
    scale_corners,
    order_corners_clockwise,
    min_corner_matching_error,
    extract_corners_from_mask
)
from paper_detection.segmentation.infer import SegmentationInference, detect_paper_corners
from paper_detection.segmentation.train import train

__all__ = [
    # Config
    'SegmentationConfig',

    # Model
    'UNet',
    'create_unet',
    'CombinedLoss',
    'DiceLoss',

    # Dataset
    'SegmentationDataset',
    'create_dataloaders',
    'generate_mask_from_corners',

    # Postprocessing
    'mask_to_corners',
    'scale_corners',
    'order_corners_clockwise',
    'min_corner_matching_error',
    'extract_corners_from_mask',

    # Inference
    'SegmentationInference',
    'detect_paper_corners',

    # Training
    'train',
]
