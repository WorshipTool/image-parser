"""
Paper detector for images
"""

from pathlib import Path
from typing import Optional

import numpy as np

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

    def detect(self, image: np.ndarray, debug: bool = False) -> dict:
        """
        Detect paper with shouldCrop decision.

        This method analyzes the heatmap and decides whether the image contains
        a physical paper document that should be cropped, or if it's a screenshot
        or other non-paper image.

        Args:
            image: Input image (BGR format)
            debug: If True, print debug information and metrics

        Returns:
            dict with:
                - shouldCrop: bool - True if paper detected and should be cropped
                - corners: Optional[np.ndarray] - [4, 2] corners or None
                - debug: dict - detailed metrics from heatmap analysis and validation

        Example:
            >>> detector = PaperDetector()
            >>> result = detector.detect(image, debug=True)
            >>> if result['shouldCrop']:
            ...     corners = result['corners']
            ...     warped = warp_paper(image, corners)
            >>> else:
            ...     print(f"Not a paper photo: {result['debug']['heatmap_analysis']['rejection_reason']}")
        """
        try:
            return self.detector.detect_paper_corners(image, debug=debug)
        except Exception as e:
            print(f"Error detecting paper: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return {
                'shouldCrop': False,
                'corners': None,
                'debug': {'error': str(e)}
            }
