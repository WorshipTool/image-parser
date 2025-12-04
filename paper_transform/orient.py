"""
Auto-orientation for warped paper documents
"""

import cv2
import numpy as np
from typing import Literal


def auto_orient(
    image_bgr: np.ndarray,
    orientation: Literal["auto", "portrait", "landscape"] = "auto"
) -> np.ndarray:
    """
    Automatically orient a document image to the correct reading direction.

    This function uses a simple heuristic based on aspect ratio to determine
    if the document should be in portrait or landscape orientation. It assumes
    most documents are taller than they are wide (portrait).

    For more sophisticated orientation detection, OCR-based or ML-based methods
    can be added in the future.

    Args:
        image_bgr: Input warped document image in BGR format
        orientation: Desired orientation mode:
            - "auto": Automatically detect best orientation (default)
            - "portrait": Force portrait orientation (height > width)
            - "landscape": Force landscape orientation (width > height)

    Returns:
        Oriented image in BGR format

    Example:
        >>> from paper_transform import warp_paper, auto_orient
        >>>
        >>> warped = warp_paper(image, corners)
        >>> oriented = auto_orient(warped)
        >>> cv2.imwrite('oriented.jpg', oriented)

    Note:
        The auto mode uses a simple aspect ratio heuristic:
        - If width > height: rotate 90° counterclockwise to make portrait
        - If height >= width: keep as is (already portrait)

        This works well for most documents but may need manual override
        for square documents or documents that are intentionally landscape.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is None or empty")

    height, width = image_bgr.shape[:2]

    if orientation == "auto":
        # Simple heuristic: prefer portrait orientation
        # Most documents (A4, letter, etc.) are taller than wide
        if width > height:
            # Currently landscape, rotate to portrait
            return _rotate_90_ccw(image_bgr)
        else:
            # Already portrait or square, keep as is
            return image_bgr

    elif orientation == "portrait":
        # Force portrait (height > width)
        if width > height:
            return _rotate_90_ccw(image_bgr)
        else:
            return image_bgr

    elif orientation == "landscape":
        # Force landscape (width > height)
        if height > width:
            return _rotate_90_cw(image_bgr)
        else:
            return image_bgr

    else:
        raise ValueError(
            f"Invalid orientation: {orientation}. "
            "Must be 'auto', 'portrait', or 'landscape'"
        )


def _rotate_90_ccw(image: np.ndarray) -> np.ndarray:
    """
    Rotate image 90 degrees counter-clockwise.

    Args:
        image: Input image

    Returns:
        Rotated image
    """
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _rotate_90_cw(image: np.ndarray) -> np.ndarray:
    """
    Rotate image 90 degrees clockwise.

    Args:
        image: Input image

    Returns:
        Rotated image
    """
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def rotate_180(image: np.ndarray) -> np.ndarray:
    """
    Rotate image 180 degrees.

    This is a utility function that can be used when the document is upside down.

    Args:
        image: Input image

    Returns:
        Rotated image

    Example:
        >>> from paper_transform import auto_orient, rotate_180
        >>>
        >>> oriented = auto_orient(warped)
        >>> if document_is_upside_down:  # manual check
        >>>     oriented = rotate_180(oriented)
    """
    return cv2.rotate(image, cv2.ROTATE_180)
