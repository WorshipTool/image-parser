"""
Paper detector for images
"""

import numpy as np
from typing import Optional
from pathlib import Path

from paper_detection.types import Corners
from paper_detection.model.infer import SegmentationInference


class PaperDetector:
    """
    Class for paper detection in images using U-Net based segmentation.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the detector.

        Args:
            model_path: Path to model weights (optional, uses default if None)
        """
        self.detector = SegmentationInference(model_path=model_path)

    def detect(self, image: np.ndarray, debug: bool = False) -> Optional[Corners]:
        """
        Detect paper in the image.

        Args:
            image: Input image (BGR format)
            debug: If True, print debug information

        Returns:
            Array with 4 paper corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            or None if paper was not found.
            Corners are ordered clockwise starting from corner closest to top-left
        """
        try:
            corners = self.detector.detect_corners(image, debug=debug)
            return corners
        except Exception as e:
            print(f"Error detecting paper: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return None
