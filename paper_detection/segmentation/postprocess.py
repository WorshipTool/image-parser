"""
Post-processing for extracting corners from segmentation masks
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def order_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    """
    Order corners in clockwise direction based on angle from centroid.

    The first corner will be the one closest to top-left corner of image (0, 0).
    This ensures consistent ordering regardless of paper rotation or perspective.

    Args:
        corners: Array of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        Ordered corners in clockwise direction starting from the corner closest to top-left
    """
    if len(corners) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corners)}")

    # Calculate centroid (center point)
    centroid = np.mean(corners, axis=0)

    # Calculate angle from centroid to each corner
    # Using atan2(y - cy, x - cx) where (cx, cy) is centroid
    # atan2 returns angles in range [-pi, pi]
    angles = np.arctan2(
        corners[:, 1] - centroid[1],  # dy
        corners[:, 0] - centroid[0]   # dx
    )

    # Sort by angle (clockwise order)
    sorted_indices = np.argsort(angles)
    sorted_corners = corners[sorted_indices]

    # Find which corner is closest to top-left (0, 0)
    distances_to_origin = np.sum(sorted_corners ** 2, axis=1)  # squared distance from (0,0)
    closest_idx = np.argmin(distances_to_origin)

    # Rotate array so closest corner is first
    ordered_corners = np.roll(sorted_corners, -closest_idx, axis=0)

    return ordered_corners


def clean_mask(mask: np.ndarray, morph_kernel_size: int = 5) -> np.ndarray:
    """
    Clean binary mask using morphological operations

    Args:
        mask: Binary mask (0 or 255), shape [H, W]
        morph_kernel_size: Size of morphological kernel

    Returns:
        Cleaned binary mask
    """
    # Create kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

    # Close small holes
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Open to remove small noise
    mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    return mask_cleaned


def extract_corners_from_mask(
    mask: np.ndarray,
    min_contour_area: int = 1000,
    approx_epsilon: float = 0.02,
    debug: bool = False
) -> Optional[np.ndarray]:
    """
    Extract 4 paper corners from binary segmentation mask

    Process:
    1. Clean mask with morphological operations
    2. Find contours
    3. Select largest contour
    4. Approximate polygon to get 4 corners
    5. Order corners clockwise

    Args:
        mask: Binary mask (0 or 255), shape [H, W]
        min_contour_area: Minimum area for valid contour
        approx_epsilon: Polygon approximation epsilon (fraction of perimeter)
        debug: If True, print debug information

    Returns:
        Array with 4 corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] or None if failed
    """
    # Clean mask
    mask_cleaned = clean_mask(mask)

    # Find contours
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        if debug:
            print("No contours found in mask")
        return None

    # Select largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < min_contour_area:
        if debug:
            print(f"Largest contour area {area:.0f} is below threshold {min_contour_area}")
        return None

    # Try to approximate polygon to get 4 corners
    perimeter = cv2.arcLength(largest_contour, True)
    epsilon = approx_epsilon * perimeter
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if debug:
        print(f"Contour area: {area:.0f}, perimeter: {perimeter:.0f}, approx points: {len(approx)}")

    # If we got exactly 4 points, use them
    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
        return order_corners_clockwise(corners)

    # If we got more than 4 points, try different strategies
    if len(approx) > 4:
        # Strategy 1: Try tighter approximation
        for epsilon_mult in [1.5, 2.0, 2.5, 3.0]:
            epsilon_tight = epsilon * epsilon_mult
            approx_tight = cv2.approxPolyDP(largest_contour, epsilon_tight, True)
            if len(approx_tight) == 4:
                corners = approx_tight.reshape(4, 2).astype(np.float32)
                return order_corners_clockwise(corners)

        # Strategy 2: Use convex hull + minAreaRect
        hull = cv2.convexHull(largest_contour)
        rect = cv2.minAreaRect(hull)
        corners = cv2.boxPoints(rect).astype(np.float32)
        return order_corners_clockwise(corners)

    # If we got fewer than 4 points, use minAreaRect
    if len(approx) < 4:
        rect = cv2.minAreaRect(largest_contour)
        corners = cv2.boxPoints(rect).astype(np.float32)
        return order_corners_clockwise(corners)

    return None


def mask_to_corners(
    mask_prob: np.ndarray,
    threshold: float = 0.5,
    min_contour_area: int = 1000,
    approx_epsilon: float = 0.02,
    debug: bool = False
) -> Optional[np.ndarray]:
    """
    Convert probability mask to 4 paper corners

    This is the main entry point for post-processing.

    Args:
        mask_prob: Probability mask [H, W] with values in [0, 1]
        threshold: Threshold for binarization
        min_contour_area: Minimum area for valid contour
        approx_epsilon: Polygon approximation epsilon
        debug: If True, print debug information

    Returns:
        Array with 4 corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] or None if failed
    """
    # Threshold to binary mask
    mask_binary = (mask_prob > threshold).astype(np.uint8) * 255

    # Extract corners
    corners = extract_corners_from_mask(
        mask_binary,
        min_contour_area=min_contour_area,
        approx_epsilon=approx_epsilon,
        debug=debug
    )

    return corners


def scale_corners(corners: np.ndarray, from_size: Tuple[int, int], to_size: Tuple[int, int]) -> np.ndarray:
    """
    Scale corners from one image size to another

    Args:
        corners: Corners in source size [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        from_size: Source size (width, height)
        to_size: Target size (width, height)

    Returns:
        Scaled corners
    """
    from_w, from_h = from_size
    to_w, to_h = to_size

    scale_x = to_w / from_w
    scale_y = to_h / from_h

    scaled_corners = corners.copy()
    scaled_corners[:, 0] *= scale_x
    scaled_corners[:, 1] *= scale_y

    return scaled_corners


def min_corner_matching_error(pred_corners: np.ndarray, gt_corners: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute minimal corner matching error considering all cyclic rotations and mirroring

    This handles the fact that:
    1. Corners may be in different cyclic order (4 rotations)
    2. Corners may be in reversed order (CW vs CCW) (2 options)

    Total: 4 rotations × 2 orderings = 8 possible matchings

    Args:
        pred_corners: Predicted corners [4, 2]
        gt_corners: Ground truth corners [4, 2]

    Returns:
        Tuple of (min_error, best_matched_pred_corners)
        - min_error: Minimal mean L2 error
        - best_matched_pred_corners: Predicted corners reordered to best match GT
    """
    min_error = float('inf')
    best_pred = None

    # Try all 4 cyclic rotations
    for rotation in range(4):
        rotated = np.roll(pred_corners, rotation, axis=0)

        # Try normal order
        error_normal = np.mean(np.linalg.norm(rotated - gt_corners, axis=1))
        if error_normal < min_error:
            min_error = error_normal
            best_pred = rotated.copy()

        # Try reversed order (CW vs CCW)
        reversed_corners = rotated[::-1]
        error_reversed = np.mean(np.linalg.norm(reversed_corners - gt_corners, axis=1))
        if error_reversed < min_error:
            min_error = error_reversed
            best_pred = reversed_corners.copy()

    return min_error, best_pred
