"""
ML inference wrapper for paper corner detection.

This module provides a high-level interface for using a trained CNN model to detect
paper corners in images. The API is designed to be compatible with the OpenCV-based
detector, making it easy to swap between traditional computer vision and ML approaches.

Key Features:
    - Compatible API with OpenCV detector
    - Automatic device selection (GPU/CPU)
    - Input validation and error handling
    - Corner validation to ensure reasonable predictions
    - Graceful fallback to CPU if CUDA runs out of memory

Coordinate System:
    - Input: BGR image (OpenCV format) of any size
    - Output: 4 corners as [[x, y], [x, y], [x, y], [x, y]] in pixel coordinates
    - Corners are in the same coordinate system as the input image
    - (0, 0) is top-left corner of image

Example Usage:
    >>> detector = MLPaperDetector()
    >>> image = cv2.imread("test.jpg")
    >>> corners = detector.predict_corners(image)
    >>> if corners is not None:
    ...     print(f"Detected corners: {corners}")
    ...     # Draw corners on image
    ...     for corner in corners:
    ...         cv2.circle(image, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
    ...     cv2.imshow("Detected Corners", image)
"""

import torch
import numpy as np
import cv2
import os
import sys
from PIL import Image
from typing import Optional, Tuple

from .config import InferenceConfig
from .model import CornerDetectorCNN
from .augmentations import get_inference_transform


class MLPaperDetector:
    """
    ML-based paper corner detector using a trained CNN model.

    This class provides a clean interface for loading a trained model and using it
    to predict paper corners in images. The API is designed to match the OpenCV
    detector interface for easy interchangeability.

    Attributes:
        config: InferenceConfig object with model path and settings
        model: Loaded CornerDetectorCNN model in evaluation mode
        transform: Transform pipeline for preprocessing images
        device: torch.device for inference (CPU or CUDA)

    Example:
        >>> # Basic usage
        >>> detector = MLPaperDetector()
        >>> image = cv2.imread("test.jpg")
        >>> corners = detector.predict_corners(image)

        >>> # Custom configuration
        >>> config = InferenceConfig(
        ...     model_path="custom_model.pth",
        ...     device="cpu"
        ... )
        >>> detector = MLPaperDetector(config=config)
    """

    def __init__(self, config: InferenceConfig = None):
        """
        Initialize the ML paper detector.

        Args:
            config: InferenceConfig object with model settings. If None, uses
                   default configuration with model path at
                   'paper_detection/models/paper_detector_cnn.pth'

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        # Use default config if not provided
        self.config = config if config is not None else InferenceConfig()

        # Set device
        self.device = torch.device(self.config.device)

        # Initialize model
        self.model = CornerDetectorCNN(dropout=0.5)
        self.model.to(self.device)

        # Load trained weights
        self.load_model()

        # Set model to evaluation mode
        self.model.eval()

        # Get transform pipeline (resize + normalize)
        self.transform = get_inference_transform(self.config.image_size)

        print(f"MLPaperDetector initialized on device: {self.device}")

    def load_model(self):
        """
        Load trained model weights from checkpoint file.

        This method loads the model state dictionary from the path specified in
        config.model_path. It handles missing files gracefully with clear error
        messages.

        Raises:
            FileNotFoundError: If model checkpoint file doesn't exist
            RuntimeError: If loading the checkpoint fails

        Note:
            The checkpoint should contain a 'model_state_dict' key with the
            trained model parameters. If your checkpoint format is different,
            you may need to adjust the loading code.
        """
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.config.model_path}\n"
                f"Please train a model first or provide a valid model path.\n"
                f"To train a model, run: python -m paper_detection.ml.train"
            )

        try:
            # Load checkpoint
            checkpoint = torch.load(
                self.config.model_path,
                map_location=self.device
            )

            # Load model state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Successfully loaded model from: {self.config.model_path}")
                if 'epoch' in checkpoint:
                    print(f"Model was trained for {checkpoint['epoch']} epochs")
                if 'val_loss' in checkpoint:
                    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
            else:
                # Handle case where checkpoint is just the state dict
                self.model.load_state_dict(checkpoint)
                print(f"Successfully loaded model from: {self.config.model_path}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {self.config.model_path}: {str(e)}\n"
                f"The checkpoint file may be corrupted or incompatible."
            ) from e

    def predict_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict paper corners in an image.

        This method takes a BGR image (OpenCV format) of any size, processes it
        through the neural network, and returns corner coordinates in pixel space.

        Process:
            1. Convert BGR to RGB
            2. Store original image dimensions
            3. Resize to 224x224 (model input size)
            4. Apply normalization (ImageNet stats)
            5. Forward pass through model
            6. Denormalize corners to original image size
            7. Validate corner predictions
            8. Return corners or None if invalid

        Args:
            image: Input image as numpy array in BGR format (OpenCV convention).
                  Can be any size, will be automatically resized for inference.

        Returns:
            Numpy array of shape [4, 2] containing corner coordinates [[x, y], ...]
            in pixel coordinates of the original image. Returns None if prediction
            fails validation or if an error occurs.

            Corner order is not guaranteed - the model outputs corners in an
            arbitrary order. If you need a specific order (e.g., clockwise from
            top-left), you'll need to sort them after prediction.

        Example:
            >>> detector = MLPaperDetector()
            >>> image = cv2.imread("document.jpg")
            >>> corners = detector.predict_corners(image)
            >>> if corners is not None:
            ...     print(f"Detected corners at: {corners}")
            ... else:
            ...     print("Failed to detect valid corners")

        Note:
            - Input must be a valid numpy array with shape [H, W, 3]
            - Returns None if corners are outside image bounds
            - Returns None if corners form a degenerate quadrilateral
            - Handles CUDA out of memory by falling back to CPU
        """
        try:
            # Validate input
            if image is None or len(image.shape) != 3:
                print("Invalid image: must be a 3-channel numpy array")
                return None

            # Store original dimensions
            original_height, original_width = image.shape[:2]

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to model input size (224x224)
            # Note: transform pipeline handles this, but we do it explicitly
            # to keep track of the size for coordinate denormalization
            image_resized = cv2.resize(
                image_rgb,
                (self.config.image_size, self.config.image_size)
            )

            # Apply transform (converts to PIL, normalizes, converts to tensor)
            image_tensor = self.transform(image_resized)

            # Add batch dimension [3, H, W] -> [1, 3, H, W]
            image_tensor = image_tensor.unsqueeze(0)

            # Move to device
            image_tensor = image_tensor.to(self.device)

            # Forward pass (no gradient computation needed)
            with torch.no_grad():
                predictions = self.model(image_tensor)

            # Remove batch dimension [1, 8] -> [8]
            predictions = predictions.squeeze(0)

            # Move to CPU and convert to numpy
            corners_normalized = predictions.cpu().numpy()

            # Reshape from [8] to [4, 2]
            corners_normalized = corners_normalized.reshape(4, 2)

            # Denormalize corners: multiply by original image dimensions
            # Model outputs normalized coords [0, 1], we need pixel coords
            corners_pixels = corners_normalized.copy()
            corners_pixels[:, 0] *= original_width   # x coordinates
            corners_pixels[:, 1] *= original_height  # y coordinates

            # Validate corners
            if not self.validate_corners(corners_pixels, (original_height, original_width)):
                print("Predicted corners failed validation")
                return None

            return corners_pixels

        except RuntimeError as e:
            # Handle CUDA out of memory error
            if "out of memory" in str(e).lower():
                print("CUDA out of memory, falling back to CPU...")
                self.device = torch.device("cpu")
                self.model.to(self.device)
                self.config.device = "cpu"
                # Retry prediction on CPU
                return self.predict_corners(image)
            else:
                print(f"Runtime error during prediction: {str(e)}")
                return None

        except Exception as e:
            print(f"Error during corner prediction: {str(e)}")
            return None

    def validate_corners(self, corners: np.ndarray, image_shape: Tuple[int, int]) -> bool:
        """
        Validate predicted corners to ensure they form a reasonable quadrilateral.

        This method performs several checks to ensure the predicted corners are
        physically plausible and within image bounds.

        Validation checks:
            1. All corners are within image bounds (with small tolerance)
            2. No corners are NaN or infinite
            3. Quadrilateral area is above minimum threshold (100 pixels)
            4. Corners don't form a degenerate shape (near-zero area)

        Args:
            corners: Numpy array of shape [4, 2] containing corner coordinates
                    in pixel space [[x, y], [x, y], [x, y], [x, y]]
            image_shape: Tuple of (height, width) of the original image

        Returns:
            True if corners pass all validation checks, False otherwise

        Note:
            A small tolerance is allowed for corners slightly outside image bounds
            to handle edge cases where the paper extends slightly beyond the image.
            The tolerance is set to 10% of image dimensions.

        Example:
            >>> corners = np.array([[10, 20], [100, 20], [100, 150], [10, 150]])
            >>> is_valid = detector.validate_corners(corners, (200, 120))
            >>> print(is_valid)  # True
        """
        height, width = image_shape

        # Check for NaN or infinite values
        if not np.all(np.isfinite(corners)):
            print("Validation failed: corners contain NaN or infinite values")
            return False

        # Allow small tolerance for corners slightly outside bounds
        # (e.g., 10% of image dimensions)
        tolerance_x = width * 0.1
        tolerance_y = height * 0.1

        # Check if corners are within bounds (with tolerance)
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]

        if (np.any(x_coords < -tolerance_x) or np.any(x_coords > width + tolerance_x) or
            np.any(y_coords < -tolerance_y) or np.any(y_coords > height + tolerance_y)):
            print("Validation failed: corners outside image bounds")
            return False

        # Calculate area using Shoelace formula to check for degenerate shapes
        # Area = 0.5 * |sum of (x_i * y_{i+1} - x_{i+1} * y_i)|
        x = corners[:, 0]
        y = corners[:, 1]
        area = 0.5 * np.abs(
            np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        )

        # Minimum area threshold (100 pixels squared)
        # This prevents detection of very small or degenerate quadrilaterals
        min_area = 100.0

        if area < min_area:
            print(f"Validation failed: area too small ({area:.2f} < {min_area})")
            return False

        return True

    def get_confidence(self) -> float:
        """
        Get confidence score for the last prediction.

        TODO: Implement confidence score based on prediction uncertainty

        This is a placeholder method that currently returns 1.0. In the future,
        this should be enhanced to provide a meaningful confidence estimate based on:
        - Model uncertainty (e.g., Monte Carlo dropout)
        - Output variance
        - Distance from training distribution
        - Consistency with geometric constraints

        Returns:
            Confidence score in range [0, 1]. Currently always returns 1.0.

        Example:
            >>> detector = MLPaperDetector()
            >>> corners = detector.predict_corners(image)
            >>> confidence = detector.get_confidence()
            >>> print(f"Confidence: {confidence:.2%}")
        """
        # TODO: Implement confidence score based on prediction uncertainty
        # Potential approaches:
        # 1. Monte Carlo Dropout: Run multiple forward passes with dropout enabled
        #    and measure variance in predictions
        # 2. Ensemble predictions: Use multiple models and measure agreement
        # 3. Softmax temperature: Add a classification head and use softmax scores
        # 4. Reconstruction loss: Add an autoencoder branch and measure
        #    reconstruction quality
        # 5. Geometric consistency: Measure how well corners satisfy physical
        #    constraints (e.g., convexity, aspect ratio)
        return 1.0


# TODO: Implement batch prediction for multiple images
# When processing multiple images, it would be more efficient to batch them together
# for GPU inference. This would involve:
# 1. Accepting a list of images as input
# 2. Preprocessing all images and stacking them into a batch tensor
# 3. Running a single forward pass for the entire batch
# 4. Splitting predictions back to individual images
# 5. Denormalizing each prediction to its original image size
# This could provide 2-3x speedup for bulk processing.

# TODO: Add visualization method for debugging
# It would be helpful to have a built-in method to visualize predictions:
# def visualize_prediction(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
#     """Draw predicted corners and quadrilateral on image for debugging."""
#     # Draw corners as circles
#     # Draw lines connecting corners
#     # Add corner labels (1, 2, 3, 4)
#     # Return annotated image
# This would make it easier to debug and evaluate model performance.
