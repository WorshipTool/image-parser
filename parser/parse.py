"""
Main parser for extracting sheet components from images

Integrates:
- Paper detection (shouldCrop decision)
- Paper transformation (perspective correction)
- Document orientation (text-based rotation)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union

from paper_detection import PaperDetector
from paper_transform import warp_paper
from paper_transform.document_orient import orient_by_text


def get_sheet_components_from_image(
    image: Union[str, Path, np.ndarray],
    debug: bool = False
) -> list:
    """
    Extract sheet images from an input image.

    Pipeline:
    1. Paper detection - check if image contains physical paper (shouldCrop)
    2. If shouldCrop = True: crop, transform (warp), and orient the paper
    3. If shouldCrop = False: return original image as-is
    4. Return list of sheet images

    Args:
        image: Input image as file path (str/Path) or numpy array (BGR format)
        debug: If True, print debug information

    Returns:
        list: List of sheet images (np.ndarray), always contains 1 image.
              - If paper detected: warped and oriented sheet
              - If no paper: original image unchanged

    Examples:
        >>> # Photo with paper - returns transformed sheet
        >>> sheets = get_sheet_components_from_image("photo.jpg")
        >>> cv2.imwrite("output.jpg", sheets[0])

        >>> # Screenshot - returns original image
        >>> sheets = get_sheet_components_from_image("screenshot.png")
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

    # If no paper detected, return original image without transformation
    if not should_crop:
        if debug:
            rejection_reason = detection_result['debug']['heatmap_analysis'].get('rejection_reason', 'unknown')
            print(f"✗ No paper detected: {rejection_reason}, returning original image")
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


def get_sheet_components_batch(
    image_paths: list,
    debug: bool = False,
    output_dir: Optional[Path] = None
) -> list:
    """
    Process multiple images in batch.

    Args:
        image_paths: List of image file paths
        debug: If True, print debug information for each image
        output_dir: Optional directory to save processed images

    Returns:
        List of all sheet images from all input images

    Example:
        >>> paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        >>> all_sheets = get_sheet_components_batch(paths, output_dir=Path("output"))
        >>> print(f"Extracted {len(all_sheets)} sheets from {len(paths)} images")
    """
    all_sheets = []

    for i, image_path in enumerate(image_paths):
        if debug:
            print(f"\n\nProcessing {i+1}/{len(image_paths)}: {Path(image_path).name}")

        sheets = get_sheet_components_from_image(image_path, debug=debug)
        all_sheets.extend(sheets)

        # Save output if directory provided
        if output_dir and sheets:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for j, sheet in enumerate(sheets):
                output_filename = Path(image_path).stem + f"_sheet{j}.jpg"
                output_path = output_dir / output_filename
                cv2.imwrite(str(output_path), sheet)

                if debug:
                    print(f"Saved to: {output_path}")

    return all_sheets


if __name__ == "__main__":
    """
    Example usage and testing
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse.py <image_path> [--debug]")
        sys.exit(1)

    image_path = sys.argv[1]
    debug = "--debug" in sys.argv

    sheets = get_sheet_components_from_image(image_path, debug=debug)

    # Save output
    output_path = Path(image_path).stem + "_processed.jpg"
    cv2.imwrite(output_path, sheets[0])
    print(f"\n✓ Extracted {len(sheets)} sheet(s)")
    print(f"  Saved to: {output_path}")
