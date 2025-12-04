"""
Configuration dataclasses for training and inference of paper detection models.

This module provides configuration classes that encapsulate all parameters needed
for training and inference workflows.
"""

from dataclasses import dataclass
import torch


@dataclass
class TrainingConfig:
    """
    Configuration for training the paper detection model.

    This class contains all hyperparameters and settings required for the training
    process, including model architecture parameters, optimization settings, and
    data handling configuration.
    """

    # Input image dimensions (height and width will be resized to this)
    image_size: int = 256  # Increased from 224 for better feature extraction

    # Number of samples per training batch (small for small dataset)
    batch_size: int = 4  # Reduced to allow better generalization with small dataset

    # Total number of training epochs
    num_epochs: int = 200

    # Initial learning rate for optimizer (lower for pretrained ResNet18)
    learning_rate: float = 5e-4  # Reduced from 1e-3 for fine-tuning pretrained model

    # L2 regularization coefficient to prevent overfitting
    weight_decay: float = 0.01

    # Fraction of dataset used for training (remainder used for validation)
    train_split: float = 0.8

    # Random seed for reproducibility
    seed: int = 42

    # Number of epochs without improvement before stopping training
    early_stopping_patience: int = 40  # Increased patience for small dataset

    # Number of epochs without improvement before reducing learning rate
    lr_scheduler_patience: int = 15  # Increased patience

    # Factor by which learning rate is reduced (new_lr = lr * factor)
    lr_scheduler_factor: float = 0.5

    # Dropout probability for regularization in fully connected layers
    dropout: float = 0.5

    # Augmentation strength for training ('light', 'medium', 'strong')
    augmentation_strength: str = 'strong'  # Aggressive augmentation for small dataset

    # Directory where trained models will be saved
    save_dir: str = "paper_detection/models"

    # Device for training (automatically selects GPU if available)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class InferenceConfig:
    """
    Configuration for running inference with a trained paper detection model.

    This class contains settings needed to load and run a trained model on new
    images for prediction.
    """

    # Input image dimensions (must match training configuration)
    image_size: int = 224

    # Path to the trained model checkpoint file
    model_path: str = "paper_detection/models/paper_detector_cnn.pth"

    # Device for inference (automatically selects GPU if available)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Minimum confidence threshold for predictions (reserved for future use)
    confidence_threshold: float = 0.0
