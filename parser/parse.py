"""
Main parser for extracting sheet components from images

Integrates:
- Paper detection (shouldCrop decision)
- Paper transformation (perspective correction)
- Document orientation (text-based rotation)
- Sheet detection (YOLO-based detection for screenshots)
"""

import cv2
import numpy as np
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

# Add current directory to path for submodules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_detection import PaperDetector
from paper_transform import warp_paper
from paper_transform.document_orient import orient_by_text
from sheet_detection import detect_simple  # Auto-initializes model on import


def get_sheet_components_from_image(
    image: Union[str, Path, np.ndarray],
    debug: bool = False
) -> list:
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

    # If no paper detected, likely a screenshot - use sheet detection
    if not should_crop:
        if debug:
            rejection_reason = detection_result['debug']['heatmap_analysis'].get('rejection_reason', 'unknown')
            print(f"✗ No paper detected: {rejection_reason}")
            print(f"→ Trying sheet detection (likely screenshot)...")

        # Save image to temp file for sheet detection
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        try:
            os.close(temp_fd)
            cv2.imwrite(temp_path, image_bgr)

            # Run sheet detection
            sheet_groups = detect_simple(temp_path, show=False)

            if debug:
                print(f"  Found {len(sheet_groups)} sheet group(s)")

            # Extract sheet images from detection results
            sheet_images = []
            for i, group in enumerate(sheet_groups):
                # Try to get sheet image, fallback to group.image if sheet not available
                if group.sheet and group.sheet.image is not None:
                    sheet_images.append(group.sheet.image)
                    if debug:
                        print(f"  ✓ Extracted sheet {i+1} from .sheet")
                elif group.image is not None:
                    sheet_images.append(group.image)
                    if debug:
                        print(f"  ✓ Extracted sheet {i+1} from .image")

            # If found sheets, return them; otherwise return original image
            if sheet_images:
                if debug:
                    print(f"✓ Returning {len(sheet_images)} detected sheet(s)")
                return sheet_images
            else:
                if debug:
                    print(f"⚠ No sheets detected, returning original image")
                return [image_bgr]

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

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



if __name__ == "__main__":
    """
    Command line interface for parser
    """
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract sheet music from images using intelligent detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with debug info
  python parser/parse.py image.jpg --debug

  # Multiple images
  python parser/parse.py img1.jpg img2.jpg img3.jpg

  # Save to custom directory
  python parser/parse.py *.jpg -o output/

  # Process all JPGs in directory
  python parser/parse.py input/*.jpg -o output/ --debug
        """
    )

    parser.add_argument(
        'images',
        nargs='+',
        help='Input image path(s) - supports glob patterns'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='.',
        help='Output directory (default: current directory)'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Print debug information'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    total_sheets = 0
    for image_path in args.images:
        image_path = Path(image_path)

        if not image_path.exists():
            print(f"✗ File not found: {image_path}")
            continue

        if args.debug:
            print(f"\n{'='*60}")
            print(f"Processing: {image_path.name}")
            print('='*60)

        sheets = get_sheet_components_from_image(str(image_path), debug=args.debug)

        # Save sheets
        for i, sheet in enumerate(sheets):
            if len(sheets) == 1:
                output_filename = f"{image_path.stem}_sheet.jpg"
            else:
                output_filename = f"{image_path.stem}_sheet{i+1}.jpg"

            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), sheet)

            if args.debug:
                print(f"✓ Saved: {output_path}")

        total_sheets += len(sheets)

    # Summary
    print(f"\n{'='*60}")
    print(f"✓ Processed {len(args.images)} image(s)")
    print(f"✓ Extracted {total_sheets} sheet(s)")
    print(f"✓ Output directory: {output_dir.absolute()}")
    print('='*60)
