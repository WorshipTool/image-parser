"""
Paper detector for images
"""

import numpy as np
from typing import Optional

from paper_detection.model.detect import CornerDetector
from paper_detection.model.utils import Corners


class PaperDetector:
    """
    Class for paper detection in images using ML corner detection.
    """

    def __init__(self):
        """
        Initialize the detector with ML model.
        """
        self.corner_detector = CornerDetector()

    def detect(self, image: np.ndarray) -> Optional[Corners]:
        """
        Detect paper in the image.

        Args:
            image: Input image (BGR format)

        Returns:
            Array with 4 paper corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            or None if paper was not found.
            Corners are ordered clockwise starting from corner closest to top-left
        """
        try:
            # Detect corners using ML model
            corners = self.corner_detector.detect(image)
            return corners
        except Exception as e:
            print(f"Error detecting paper: {e}")
            return None
