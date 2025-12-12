"""
Paper detector for images
"""

import numpy as np
from typing import Optional, Literal
from pathlib import Path

from paper_detection.model.detect import CornerDetector
from paper_detection.model.utils import Corners


DetectionMode = Literal["segmentation", "regression"]


class PaperDetector:
    """
    Class for paper detection in images using ML.

    Supports two detection modes:
    - 'segmentation': U-Net based segmentation (recommended, more robust)
    - 'regression': Direct corner regression (legacy)
    """

    def __init__(self, detection_mode: DetectionMode = "segmentation", model_path: Optional[Path] = None):
        """
        Initialize the detector.

        Args:
            detection_mode: Detection mode ('segmentation' or 'regression')
            model_path: Path to model weights (optional, uses default if None)
        """
        self.detection_mode = detection_mode

        if detection_mode == "segmentation":
            # Lazy import to avoid loading dependencies if not needed
            from paper_detection.segmentation.infer import SegmentationInference
            self.detector = SegmentationInference(model_path=model_path)
        elif detection_mode == "regression":
            self.detector = CornerDetector()
        else:
            raise ValueError(f"Invalid detection_mode: {detection_mode}. Must be 'segmentation' or 'regression'")

    def detect(self, image: np.ndarray, debug: bool = False) -> Optional[Corners]:
        """
        Detect paper in the image.

        Args:
            image: Input image (BGR format)
            debug: If True, print debug information (only for segmentation mode)

        Returns:
            Array with 4 paper corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            or None if paper was not found.
            Corners are ordered clockwise starting from corner closest to top-left
        """
        try:
            if self.detection_mode == "segmentation":
                corners = self.detector.detect_corners(image, debug=debug)
            else:
                corners = self.detector.detect(image)
            return corners
        except Exception as e:
            print(f"Error detecting paper: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return None
