"""
Detector for extracting sheet components from images

Integrates:
- Paper detection (shouldCrop decision)
- Paper transformation (perspective correction)
- Document orientation (text-based rotation)
- Sheet detection (YOLO-based detection for screenshots)
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Union

import cv2
import numpy as np


# Add current directory to path for submodules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_detection import PaperDetector
from paper_transform import warp_paper
from paper_transform.document_orient import orient_by_text
from sheet_detection import detect_simple  # Auto-initializes model on import


def _detect_and_merge_sheets(image_bgr: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Detect sheets using YOLO and merge all detected bounding boxes.

    Args:
        image_bgr: Input image (BGR format)
        debug: Print debug information

    Returns:
        np.ndarray: Cropped image containing all detected sheets merged
    """
    # Save image to temp file for sheet detection
    temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
    try:
        os.close(temp_fd)
        cv2.imwrite(temp_path, image_bgr)

        # Run sheet detection
        sheet_groups = detect_simple(temp_path, show=False)

        if debug:
            print(f"  Found {len(sheet_groups)} sheet group(s)")

        if not sheet_groups:
            if debug:
                print(f"⚠ No sheets detected, returning original image")
            return image_bgr

        # Collect all sheet bounding boxes
        sheet_bounds = []
        for i, group in enumerate(sheet_groups):
            if group.sheet and group.sheet.bounds is not None:
                sheet_bounds.append(group.sheet.bounds)
                if debug:
                    bounds = group.sheet.bounds
                    print(f"  Sheet {i+1}: ({int(bounds.left)}, {int(bounds.top)}, {int(bounds.width)}, {int(bounds.height)})")

        if not sheet_bounds:
            if debug:
                print(f"⚠ No sheet bounds found, returning original image")
            return image_bgr

        # Calculate union of all bounding boxes
        min_left = min(b.left for b in sheet_bounds)
        min_top = min(b.top for b in sheet_bounds)
        max_right = max(b.left + b.width for b in sheet_bounds)
        max_bottom = max(b.top + b.height for b in sheet_bounds)

        # Crop image to union of all sheets
        cropped = image_bgr[int(min_top):int(max_bottom), int(min_left):int(max_right)]

        if debug:
            print(f"✓ Merged {len(sheet_bounds)} sheet(s) into bounding box:")
            print(f"  ({int(min_left)}, {int(min_top)}) → ({int(max_right)}, {int(max_bottom)})")
            print(f"  Size: {cropped.shape[1]}x{cropped.shape[0]}")

        return cropped

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def get_sheet_components_from_image(
    image: Union[str, Path, np.ndarray],
    debug: bool = False
) -> list[np.ndarray]:
    """
    Extract sheet images from an input image.

    Pipeline:
    1. Paper detection - check if image contains physical paper (shouldCrop)
    2. If shouldCrop = True: crop, transform (warp), and orient the paper
    3. If shouldCrop = False: use sheet detection (YOLO) to find and crop sheets
    4. Return list of sheet images

    Args:
        image: Input image as file path (str/Path) or numpy array (BGR format)
        debug: If True, print debug information

    Returns:
        list: List of sheet images (np.ndarray).
              - If paper detected: [warped and oriented sheet] (1 image)
              - If no paper (screenshot): [detected sheets...] (0+ images)
              - If no paper and no sheets found: [original image] (1 image)

    Examples:
        >>> # Photo with paper - returns transformed sheet
        >>> sheets = get_sheet_components_from_image("photo.jpg")
        >>> cv2.imwrite("output.jpg", sheets[0])

        >>> # Screenshot with multiple sheets - returns multiple cropped sheets
        >>> sheets = get_sheet_components_from_image("screenshot.png")
        >>> print(len(sheets))  # Could be 1, 2, 3+ depending on detected sheets

        >>> # Screenshot with no detectable sheets - returns original
        >>> sheets = get_sheet_components_from_image("no_sheets.png")
        >>> print(len(sheets))  # 1 (original image)
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            if debug:
                print(f"✗ Failed to load image: {image_path}")
            return []
    else:
        image_bgr = image
        if image_bgr is None or image_bgr.size == 0:
            if debug:
                print("✗ Invalid image array")
            return []

    # Step 1: Paper Detection
    detector = PaperDetector()
    detection_result = detector.detect(image_bgr, debug=debug)

    should_crop = detection_result['shouldCrop']
    corners = detection_result['corners']

    # If no paper detected, likely a screenshot - just return
    if not should_crop:
        image_bgr = _detect_and_merge_sheets(image_bgr, debug=debug)
        return [image_bgr]

    # Step 2: Perspective Correction
    try:
        warped = warp_paper(image_bgr, corners)
        if debug:
            print(f"✓ Warped to {warped.shape[1]}x{warped.shape[0]}")
    except Exception as e:
        if debug:
            print(f"✗ Warp failed: {e}")
        return []

    # Step 3: Orientation Detection
    try:
        result = orient_by_text(warped)
        if result is None:
            # No text found, return warped without rotation
            oriented = warped
            if debug:
                print("⚠ No text detected, keeping original orientation")
        else:
            oriented, rotation, confidence = result
            if debug:
                print(f"✓ Applied rotation: {rotation}° (confidence: {confidence:.2f})")
    except Exception as e:
        if debug:
            print(f"✗ Orientation failed: {e}")
        # Return warped image even if orientation fails
        oriented = warped

    # Return list with one sheet image
    return [oriented]

