"""Deep learning module for paper corner detection.

This module provides a CNN-based approach for detecting paper corners in images.

Main Components:
    - MLPaperDetector: Primary class for inference/prediction
    - CornerDetectorCNN: Neural network model architecture
    - PaperCornersDataset: Dataset class for training
    - TrainingConfig, InferenceConfig: Configuration classes

Usage Examples:
    Inference:
        >>> from paper_detection.ml import MLPaperDetector
        >>> detector = MLPaperDetector()
        >>> corners = detector.predict_corners(image)

    Training:
        >>> python -m paper_detection.ml.train
"""

from .config import TrainingConfig, InferenceConfig
from .model import CornerDetectorCNN
from .dataset import PaperCornersDataset
from .inference import MLPaperDetector
from .augmentations import get_training_augmentation, get_inference_transform

__all__ = [
    "TrainingConfig",
    "InferenceConfig",
    "CornerDetectorCNN",
    "PaperCornersDataset",
    "MLPaperDetector",
    "get_training_augmentation",
    "get_inference_transform",
]
