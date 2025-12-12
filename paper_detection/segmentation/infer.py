"""
Inference module for paper segmentation
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from paper_detection.segmentation.config import SegmentationConfig
from paper_detection.segmentation.model import UNet
from paper_detection.segmentation.postprocess import mask_to_corners, scale_corners


class SegmentationInference:
    """
    Inference class for paper segmentation

    Handles model loading, preprocessing, inference, and postprocessing
    """

    def __init__(self, model_path: Optional[Path] = None, config: Optional[SegmentationConfig] = None):
        """
        Initialize inference engine

        Args:
            model_path: Path to trained model weights
            config: Configuration (if None, uses default)
        """
        self.config = config if config is not None else SegmentationConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        if model_path is None:
            model_path = self.config.MODEL_PATH

        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: Path) -> UNet:
        """Load trained model from checkpoint"""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Create model
        model = UNet(in_channels=3, out_channels=1, bilinear=True)
        model = model.to(self.device)

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded from: {model_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        if 'val_iou' in checkpoint:
            print(f"  Val IoU: {checkpoint['val_iou']:.4f}")

        return model

    def preprocess_image(self, image_bgr: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for model input

        Args:
            image_bgr: Input image in BGR format [H, W, 3]

        Returns:
            Tuple of (preprocessed_tensor, original_size)
            - preprocessed_tensor: [1, 3, IMAGE_SIZE, IMAGE_SIZE]
            - original_size: (original_width, original_height)
        """
        # Get original size
        h, w = image_bgr.shape[:2]
        original_size = (w, h)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        image_resized = cv2.resize(
            image_rgb,
            (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize
        image_float = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_normalized = (image_float - mean) / std

        # Convert to tensor [1, 3, H, W]
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)

        return image_tensor, original_size

    def predict_mask(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict segmentation mask

        Args:
            image_bgr: Input image in BGR format [H, W, 3]

        Returns:
            Tuple of (mask_prob, mask_resized)
            - mask_prob: Probability mask [IMAGE_SIZE, IMAGE_SIZE] in range [0, 1]
            - mask_resized: Mask resized to original image size
        """
        # Preprocess
        image_tensor, original_size = self.preprocess_image(image_bgr)

        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)  # [1, 1, H, W]
            probs = torch.sigmoid(logits)      # [1, 1, H, W]

        # Convert to numpy
        mask_prob = probs[0, 0].cpu().numpy()  # [H, W]

        # Resize mask to original size
        mask_resized = cv2.resize(
            mask_prob,
            original_size,
            interpolation=cv2.INTER_LINEAR
        )

        return mask_prob, mask_resized

    def detect_corners(self, image_bgr: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
        """
        Detect paper corners from image

        Args:
            image_bgr: Input image in BGR format [H, W, 3]
            debug: If True, print debug information

        Returns:
            Corners [4, 2] in original image coordinates or None if failed
        """
        # Get original size
        h, w = image_bgr.shape[:2]
        original_size = (w, h)

        # Predict mask (at model resolution)
        mask_prob, _ = self.predict_mask(image_bgr)

        # Extract corners at model resolution
        corners = mask_to_corners(
            mask_prob,
            threshold=self.config.MASK_THRESHOLD,
            min_contour_area=self.config.MIN_CONTOUR_AREA,
            approx_epsilon=self.config.APPROX_EPSILON,
            debug=debug
        )

        if corners is None:
            return None

        # Scale corners from model resolution to original resolution
        model_size = (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        corners_scaled = scale_corners(corners, model_size, original_size)

        return corners_scaled

    def visualize_detection(
        self,
        image_bgr: np.ndarray,
        corners: Optional[np.ndarray],
        mask_prob: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Create visualization of detection result

        Args:
            image_bgr: Original image
            corners: Detected corners [4, 2]
            mask_prob: Probability mask (optional)
            output_path: Path to save visualization (optional)

        Returns:
            Visualization image
        """
        vis = image_bgr.copy()

        # Overlay mask if provided
        if mask_prob is not None:
            # Resize mask to image size if needed
            h, w = image_bgr.shape[:2]
            if mask_prob.shape != (h, w):
                mask_resized = cv2.resize(mask_prob, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                mask_resized = mask_prob

            # Convert to colormap
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
            mask_colored = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)

            # Blend with original image
            vis = cv2.addWeighted(vis, 0.6, mask_colored, 0.4, 0)

        # Draw corners if found
        if corners is not None:
            corners_int = corners.astype(np.int32)

            # Draw polygon
            cv2.polylines(vis, [corners_int], True, (0, 255, 0), 3)

            # Draw corner points with labels
            labels = ['TL', 'TR', 'BR', 'BL']
            for idx, corner in enumerate(corners_int):
                cv2.circle(vis, tuple(corner), 8, (0, 255, 0), -1)
                cv2.putText(
                    vis, labels[idx], tuple(corner + np.array([10, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

        # Save if output path provided
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis)

        return vis


def detect_paper_corners(image_bgr: np.ndarray, model_path: Optional[Path] = None) -> Optional[np.ndarray]:
    """
    Convenience function to detect paper corners

    Args:
        image_bgr: Input image in BGR format
        model_path: Path to model weights (optional)

    Returns:
        Corners [4, 2] or None if failed
    """
    inference = SegmentationInference(model_path=model_path)
    corners = inference.detect_corners(image_bgr)
    return corners
