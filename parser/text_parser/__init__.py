"""
Text Parser Module

OCR-based text extraction and chord parsing for sheet music.
Takes an image and extracts structured text with chord positions.

No preprocessing - just reads the image as-is.
"""

import numpy as np
import sys
from typing import Optional, Union
from pathlib import Path
import cv2

from parser.text_parser.preprocess import preprocess

# Add parent directories to path for common module access
_current_dir = Path(__file__).parent
_parser_dir = _current_dir.parent
_image_parser_dir = _parser_dir.parent
sys.path.insert(0, str(_image_parser_dir))

from .ocr import read as ocr_read
from .formatter import format as format_sheet


def read_and_parse_image(
    image: Union[str, Path, np.ndarray],
    debug: bool = False
) -> Optional[dict]:
    """
    Read text from image using OCR and parse into structured sheet format.

    This function:
    1. Runs OCR (Pytesseract) on the image
    2. Parses words into lines and sections
    3. Detects chords using regex
    4. Formats output with chord positions

    Args:
        image: Input image path or numpy array (BGR format)
        debug: Print debug information

    Returns:
        dict with:
            - title: str - Detected song title
            - data: str - Formatted text with chords (e.g., "{V1}[Am]text...")
            - inputImagePath: str - Source image path
        or None if reading failed

    Example:
        >>> result = read_and_parse_image("sheet.jpg")
        >>> print(result['title'])
        "Amazing Grace"
        >>> print(result['data'])
        "{V1}[Am]Amazing grace...\n{Chorus1}[G]How sweet..."
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image_path = str(Path(image))
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            if debug:
                print(f"✗ Failed to load image: {image_path}")
            return None
    else:
        image_bgr = image
        image_path = "unknown"
        if image_bgr is None or image_bgr.size == 0:
            if debug:
                print("✗ Invalid image array")
            return None

    if debug:
        print(f"Reading text from image: {image_bgr.shape[1]}x{image_bgr.shape[0]}")

    # Preprocess image
    image_bgr = preprocess(image_bgr, debug=debug)

    
    # Run OCR
    try:
        word_data = ocr_read(image_bgr)
        if debug:
            print(f"  ✓ OCR detected {len(word_data)} words")
    except Exception as e:
        if debug:
            print(f"✗ OCR failed: {e}")
        return None

    if not word_data:
        if debug:
            print("⚠ No text detected in image")
        return None

    # Parse and format
    try:
        # Use the same data for both title and content parsing
        # formatter.format() will extract title from first lines
        sheet = format_sheet( word_data, image_path, image_bgr, debug=debug)

        if debug:
            print(f"  ✓ Parsed title: {sheet.title}")
            print(f"  ✓ Formatted {len(sheet.data)} characters")

        return sheet.to_json()

    except Exception as e:
        if debug:
            import traceback
            print(f"✗ Formatting failed: {e}")
            traceback.print_exc()
        return None



