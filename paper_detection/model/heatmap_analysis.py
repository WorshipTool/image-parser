"""
Heatmap analysis for deciding whether to crop based on paper presence
"""

import cv2
import numpy as np
from typing import Dict, Tuple


def analyze_heatmap(
    mask_prob: np.ndarray,
    threshold: float = 0.5,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.65,  # Lowered from 0.95 to reject screenshots with large white areas
    min_component_ratio: float = 0.7,
    min_extent: float = 0.4,
    max_bbox_cover: float = 0.75,  # Lowered from 0.95 to reject fullscreen-like content
    debug: bool = False
) -> Dict:
    """
    Analyze probability heatmap to decide if paper is present

    This runs BEFORE expensive edge/line detection as a fast gate.

    Args:
        mask_prob: Probability mask [H, W] with values in [0, 1]
        threshold: Threshold for binarization (default 0.5)
        min_area_ratio: Minimum fraction of image that should be paper (default 5%)
        max_area_ratio: Maximum fraction (reject fullscreen, default 65%)
        min_component_ratio: Minimum ratio of largest component to total area (default 70%)
        min_extent: Minimum extent (component_area / bbox_area, default 40%)
        max_bbox_cover: Maximum bbox coverage of image (reject fullscreen, default 75%)
        debug: If True, print debug information

    Returns:
        dict with:
            - should_crop: bool - whether to proceed with cropping
            - metrics: dict - computed metrics for debugging
            - rejection_reason: str or None - why it was rejected
    """
    h, w = mask_prob.shape
    total_pixels = h * w

    # Step 1: Threshold to binary mask
    mask_binary = (mask_prob > threshold).astype(np.uint8) * 255

    # Step 2: Compute area ratio
    paper_pixels = np.sum(mask_binary > 0)
    area_ratio = paper_pixels / total_pixels

    if debug:
        print(f"\n=== Heatmap Analysis ===")
        print(f"Image size: {w}x{h} ({total_pixels} pixels)")
        print(f"Paper pixels: {paper_pixels} ({area_ratio:.1%})")

    # Step 3: Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=8
    )

    # Ignore background (label 0)
    if num_labels <= 1:
        if debug:
            print("No components found")
        return {
            'should_crop': False,
            'metrics': {
                'area_ratio': area_ratio,
                'num_components': 0,
            },
            'rejection_reason': 'no_components'
        }

    # Get areas of all components (excluding background)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(component_areas) + 1  # +1 because we excluded background

    largest_component_area = stats[largest_idx, cv2.CC_STAT_AREA]
    largest_component_ratio = largest_component_area / paper_pixels if paper_pixels > 0 else 0

    # Get bounding box of largest component
    bbox_x = stats[largest_idx, cv2.CC_STAT_LEFT]
    bbox_y = stats[largest_idx, cv2.CC_STAT_TOP]
    bbox_w = stats[largest_idx, cv2.CC_STAT_WIDTH]
    bbox_h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
    bbox_area = bbox_w * bbox_h

    # Compute extent (how well component fills its bounding box)
    extent = largest_component_area / bbox_area if bbox_area > 0 else 0

    # Compute bbox coverage (how much of image the bbox covers)
    bbox_cover = bbox_area / total_pixels

    metrics = {
        'area_ratio': area_ratio,
        'num_components': num_labels - 1,  # Exclude background
        'largest_component_area': largest_component_area,
        'largest_component_ratio': largest_component_ratio,
        'bbox': (bbox_x, bbox_y, bbox_w, bbox_h),
        'bbox_area': bbox_area,
        'extent': extent,
        'bbox_cover': bbox_cover,
    }

    if debug:
        print(f"Components: {num_labels - 1}")
        print(f"Largest component: {largest_component_area} pixels ({largest_component_ratio:.1%} of paper area)")
        print(f"Bbox: ({bbox_x}, {bbox_y}, {bbox_w}, {bbox_h})")
        print(f"Extent: {extent:.1%} (component fills bbox)")
        print(f"Bbox cover: {bbox_cover:.1%} (bbox covers image)")

    # Step 4: Early rejection criteria
    rejection_reason = None

    # Check 1: Area ratio too small (paper not visible)
    if area_ratio < min_area_ratio:
        rejection_reason = f'area_too_small ({area_ratio:.1%} < {min_area_ratio:.1%})'

    # Check 2: Area ratio too large (fullscreen / screenshot-like)
    elif area_ratio > max_area_ratio:
        rejection_reason = f'area_too_large ({area_ratio:.1%} > {max_area_ratio:.1%})'

    # Check 3: Largest component ratio too low (no single dominant region)
    elif largest_component_ratio < min_component_ratio:
        rejection_reason = f'fragmented ({largest_component_ratio:.1%} < {min_component_ratio:.1%})'

    # Check 4: Extent too low (shape not paper-like/rectangular)
    elif extent < min_extent:
        rejection_reason = f'extent_low ({extent:.1%} < {min_extent:.1%})'

    # Check 5: Bbox covers almost entire image (fullscreen)
    elif bbox_cover > max_bbox_cover:
        rejection_reason = f'bbox_fullscreen ({bbox_cover:.1%} > {max_bbox_cover:.1%})'

    should_crop = rejection_reason is None

    if debug:
        print(f"\n=== Decision ===")
        if should_crop:
            print("✓ Heatmap gate PASSED - proceed to corner detection")
        else:
            print(f"✗ Heatmap gate REJECTED: {rejection_reason}")

    return {
        'should_crop': should_crop,
        'metrics': metrics,
        'rejection_reason': rejection_reason
    }


def validate_corners(
    corners: np.ndarray,
    image_shape: Tuple[int, int],
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.98,
    debug: bool = False
) -> Dict:
    """
    Validate that detected corners form a reasonable quadrilateral

    Args:
        corners: [4, 2] array of corners
        image_shape: (height, width) of image
        min_area_ratio: Minimum area as fraction of image
        max_area_ratio: Maximum area as fraction of image
        debug: If True, print debug information

    Returns:
        dict with:
            - is_valid: bool
            - metrics: dict
            - rejection_reason: str or None
    """
    h, w = image_shape
    image_area = h * w

    # Compute area of quadrilateral using shoelace formula
    x = corners[:, 0]
    y = corners[:, 1]
    quad_area = 0.5 * abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    )

    area_ratio = quad_area / image_area

    # Check if corners are in valid range
    corners_in_bounds = np.all(
        (corners[:, 0] >= 0) & (corners[:, 0] < w) &
        (corners[:, 1] >= 0) & (corners[:, 1] < h)
    )

    # Check if area is reasonable
    rejection_reason = None

    if not corners_in_bounds:
        rejection_reason = 'corners_out_of_bounds'
    elif area_ratio < min_area_ratio:
        rejection_reason = f'quad_area_too_small ({area_ratio:.1%})'
    elif area_ratio > max_area_ratio:
        rejection_reason = f'quad_area_too_large ({area_ratio:.1%})'

    is_valid = rejection_reason is None

    metrics = {
        'quad_area': quad_area,
        'area_ratio': area_ratio,
        'corners_in_bounds': corners_in_bounds,
    }

    if debug:
        print(f"\n=== Corner Validation ===")
        print(f"Quad area: {quad_area:.0f} ({area_ratio:.1%} of image)")
        print(f"Corners in bounds: {corners_in_bounds}")
        if is_valid:
            print("✓ Corners VALID")
        else:
            print(f"✗ Corners INVALID: {rejection_reason}")

    return {
        'is_valid': is_valid,
        'metrics': metrics,
        'rejection_reason': rejection_reason
    }
