"""
Perspective warp transformation for paper documents
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def _order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order corners in consistent order: top-left, top-right, bottom-right, bottom-left.

    Args:
        corners: Array of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        Ordered corners array
    """
    # Already ordered by PaperDetector, but let's ensure consistency
    # In case corners come from another source

    # Convert to float32 for calculations
    pts = corners.astype(np.float32)

    # Sort by y-coordinate to get top and bottom pairs
    pts_sorted_by_y = pts[pts[:, 1].argsort()]

    # Top two points (smaller y values)
    top_pts = pts_sorted_by_y[:2]
    # Bottom two points (larger y values)
    bottom_pts = pts_sorted_by_y[2:]

    # Sort top points by x to get top-left and top-right
    top_pts = top_pts[top_pts[:, 0].argsort()]
    top_left = top_pts[0]
    top_right = top_pts[1]

    # Sort bottom points by x to get bottom-left and bottom-right
    bottom_pts = bottom_pts[bottom_pts[:, 0].argsort()]
    bottom_left = bottom_pts[0]
    bottom_right = bottom_pts[1]

    # Return in order: top-left, top-right, bottom-right, bottom-left
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _compute_output_dimensions(corners: np.ndarray) -> Tuple[int, int]:
    """
    Compute the output dimensions for the warped image based on corner distances.

    Args:
        corners: Ordered corners [top-left, top-right, bottom-right, bottom-left]

    Returns:
        Tuple of (width, height) for output image
    """
    top_left, top_right, bottom_right, bottom_left = corners

    # Compute width as the maximum of top and bottom edge lengths
    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    width = int(max(width_top, width_bottom))

    # Compute height as the maximum of left and right edge lengths
    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    height = int(max(height_left, height_right))

    return width, height


def warp_paper(
    image_bgr: np.ndarray,
    corners: np.ndarray,
    dst_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Apply perspective transformation to straighten the paper document.

    This function takes an image and the detected corners of a paper document,
    and returns a warped (straightened) view of the document using perspective
    transformation.

    Args:
        image_bgr: Input image in BGR format (from cv2.imread)
        corners: Array of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
                 Can be in any order - will be automatically ordered.
        dst_size: Optional tuple (width, height) for output image.
                  If None, automatically computed from corner distances.

    Returns:
        Warped image in BGR format with the paper document straightened

    Example:
        >>> from paper_detection import PaperDetector
        >>> from paper_transform import warp_paper
        >>>
        >>> image = cv2.imread('photo.jpg')
        >>> detector = PaperDetector()
        >>> corners = detector.detect(image)
        >>> if corners is not None:
        >>>     warped = warp_paper(image, corners)
        >>>     cv2.imwrite('warped.jpg', warped)
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is None or empty")

    if corners is None or len(corners) != 4:
        raise ValueError("Corners must be an array of 4 points")

    # Order corners consistently
    ordered_corners = _order_corners(corners)

    # Compute output dimensions if not provided
    if dst_size is None:
        width, height = _compute_output_dimensions(ordered_corners)
    else:
        width, height = dst_size

    # Define destination points for the warped output
    # These form a perfect rectangle
    dst_corners = np.array([
        [0, 0],                    # top-left
        [width - 1, 0],            # top-right
        [width - 1, height - 1],   # bottom-right
        [0, height - 1]            # bottom-left
    ], dtype=np.float32)

    # Compute perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_corners)

    # Apply perspective transformation
    warped = cv2.warpPerspective(
        image_bgr,
        transform_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # White border for paper
    )

    return warped
