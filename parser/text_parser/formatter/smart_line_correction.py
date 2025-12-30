"""
Smart line correction module

Applies intelligent corrections to detected lines before formatting.
"""
from typing import List
import numpy as np
import cv2
from pathlib import Path
from .line import Line

# Counter for unique line image filenames
_line_counter = 0


def smart_line_correction(line: Line, image: np.ndarray) -> Line:
    """
    Apply smart corrections to a single line

    Args:
        line: Line object to correct
        image: Cropped line image (BGR format)

    Returns:
        Corrected Line object
    """
    global _line_counter

    # Save line image to temp folder
    # Navigate from this file to image-parser root
    current_file = Path(__file__).resolve()
    image_parser_root = current_file.parent.parent.parent.parent
    temp_dir = image_parser_root / "temp" / "line_corrections"
    temp_dir.mkdir(parents=True, exist_ok=True)

    output_path = temp_dir / f"line_{_line_counter:04d}.jpg"
    cv2.imwrite(str(output_path), image)
    _line_counter += 1

    # TEST: Replace all word texts with "Ahoj"
    for word in line.words:
        word.text = "Ahoj"

    return line


def smart_lines_correction(lines: List[Line], image: np.ndarray) -> List[Line]:
    """
    Apply smart corrections to all lines

    Args:
        lines: List of Line objects to correct
        image_bgr: Original image (BGR format) from which text was extracted

    Returns:
        List of corrected Line objects
    """
    corrected_lines = []

    for line in lines:
        should_correct = line.avgConfidence < 94
        if should_correct:
            # Crop image to line bounds

            croped_image = image[
                int(line.bounds.top):int(line.bounds.top + line.bounds.height),
                int(line.bounds.left):int(line.bounds.left + line.bounds.width)
            ]
            corrected_line = smart_line_correction(line, croped_image) 
        else:
            corrected_line = line
        corrected_lines.append(corrected_line)

    return corrected_lines
