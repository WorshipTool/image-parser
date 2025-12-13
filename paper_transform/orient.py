"""
Auto-orientation for warped paper documents
"""

import cv2
import numpy as np
from typing import Literal, Optional
import logging

logger = logging.getLogger(__name__)


def auto_orient(
    image_bgr: np.ndarray,
    orientation: Literal["auto", "auto_text", "auto_geometric", "portrait", "landscape"] = "auto",
    use_ocr: bool = True,
    ocr_engine: str = "tesseract",
    debug: bool = False
) -> np.ndarray:
    """
    Automatically orient a document image to the correct reading direction.

    This function can use two methods:
    1. Text-based orientation (default): Uses OCR to detect text orientation
    2. Geometric orientation: Uses aspect ratio heuristic

    Args:
        image_bgr: Input warped document image in BGR format
        orientation: Desired orientation mode:
            - "auto": Try text-based first, fall back to geometric (default)
            - "auto_text": Force text-based orientation (requires OCR)
            - "auto_geometric": Force geometric orientation (aspect ratio)
            - "portrait": Force portrait orientation (height > width)
            - "landscape": Force landscape orientation (width > height)
        use_ocr: If True and orientation is "auto", try OCR-based detection first
        ocr_engine: OCR engine to use ('tesseract' or 'easyocr')
        debug: If True, print debug information

    Returns:
        Oriented image in BGR format

    Example:
        >>> from paper_transform import warp_paper, auto_orient
        >>>
        >>> # Text-based orientation (default)
        >>> warped = warp_paper(image, corners)
        >>> oriented = auto_orient(warped)
        >>>
        >>> # Force geometric orientation only
        >>> oriented = auto_orient(warped, orientation="auto_geometric")
        >>>
        >>> # Debug mode to see scores
        >>> oriented = auto_orient(warped, debug=True)

    Note:
        Text-based orientation is more accurate but slower. It tries all
        four rotations and selects the one with the most readable text.

        Geometric orientation is fast but less accurate. It uses aspect ratio
        to prefer portrait orientation.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is None or empty")

    height, width = image_bgr.shape[:2]

    # Handle text-based orientation modes
    if orientation in ["auto", "auto_text"]:
        if use_ocr or orientation == "auto_text":
            try:
                from .document_orient import orient_by_text
                return orient_by_text(
                    image_bgr,
                    ocr_engine=ocr_engine,
                    debug=debug
                )
            except ImportError as e:
                if orientation == "auto_text":
                    raise ImportError(
                        f"OCR engine not available for text-based orientation: {e}\n"
                        "Install with: pip install pytesseract"
                    )
                else:
                    # Fall through to geometric orientation
                    logger.info("OCR not available, using geometric orientation")
            except Exception as e:
                if debug:
                    logger.warning(f"Text-based orientation failed: {e}, falling back to geometric")
                # Fall through to geometric orientation

    # Handle geometric orientation modes
    if orientation in ["auto", "auto_geometric"]:
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
