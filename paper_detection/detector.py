"""
Paper detector for images
"""

import numpy as np
from typing import Optional


class PaperDetector:
    """
    Class for paper detection in images.
    """

    def __init__(self):
        """
        Initialize the detector.
        """
        pass

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect paper in the image.

        Args:
            image: Input image (BGR format)

        Returns:
            Array with 4 paper corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            or None if paper was not found.
            Corners are ordered: top-left, top-right, bottom-right, bottom-left
        """
        # TODO: Implement paper detection
        return None
