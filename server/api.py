"""
Server API for programmatic access to parser

This module provides a Python API for the parser, allowing batch processing
of images without using the CLI interface. Used by the Flask server for
job processing.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Generator, List, Optional


# Add paths for parser module access
_current_dir = Path(__file__).parent
_image_parser_root = _current_dir.parent
sys.path.insert(0, str(_image_parser_root))
sys.path.insert(0, str(_image_parser_root / "parser"))

from parser.get_sheet_components import get_sheet_components_from_image
from parser.text_parser import read_and_parse_image


def parse_images(
    image_paths: List[str],
    use_ai: bool = False,
    debug: bool = False
) -> Generator[int, None, List[Dict]]:
    """
    Parse multiple images and return structured sheet data with progress reporting.

    This function processes images in batch, yielding progress percentage
    and returning final results when complete.

    Args:
        image_paths: List of paths to input images
        use_ai: Enable AI-based corrections (line_correction and final_smart_correction)
        debug: Save debug information and intermediate files

    Yields:
        int: Progress percentage (0-100)

    Returns:
        List[Dict]: List of parsed sheets with structure:
            {
                "title": str,
                "data": str,  # Formatted text with chords
                "inputImagePath": str
            }

    Example:
        >>> gen = parse_images(["sheet1.jpg", "sheet2.jpg"], use_ai=True)
        >>> for progress in gen:
        ...     print(f"Progress: {progress}%")
        >>> # When StopIteration is raised, access result via exception value
        >>> try:
        ...     while True:
        ...         progress = next(gen)
        ... except StopIteration as e:
        ...     results = e.value
    """

    if not image_paths:
        return []

    yield 0  # 0% progress

    total_images = len(image_paths)
    all_results = []

    for idx, image_path in enumerate(image_paths):
        image_path = Path(image_path)

        if not image_path.exists():
            if debug:
                print(f"✗ File not found: {image_path}")
            continue

        # Calculate progress offset for this image
        base_progress = int((idx / total_images) * 100)
        image_progress_range = int(100 / total_images)

        def calc_progress(local_percent: float) -> int:
            """Calculate global progress from local image progress"""
            return base_progress + int((local_percent / 100) * image_progress_range)

        # Step 1: Extract sheet components (0-30% of image processing)
        yield calc_progress(0)

        sheets = get_sheet_components_from_image(str(image_path), debug=debug)

        if not sheets:
            yield calc_progress(100)
            continue

        yield calc_progress(30)

        # Step 2: Parse text from each detected sheet (30-100% of image processing)
        for sheet_idx, sheet in enumerate(sheets):
            # Progress within this image's sheets
            sheet_base = 30
            sheet_range = 70
            sheet_progress = sheet_base + int((sheet_idx / len(sheets)) * sheet_range)

            yield calc_progress(sheet_progress)

            # Parse the sheet
            text_result = read_and_parse_image(sheet, debug=debug, use_ai=use_ai)

            if text_result:
                # Use original image filename in result
                text_result['inputImagePath'] = str(image_path.name)
                all_results.append(text_result)

                if debug:
                    print(f"✓ Parsed sheet {sheet_idx + 1}/{len(sheets)}: {text_result['title']}")

        # Complete this image
        yield calc_progress(100)

    # Final progress
    yield 100

    return all_results


def parse_single_image(
    image_path: str,
    use_ai: bool = False,
    debug: bool = False
) -> Optional[Dict]:
    """
    Parse a single image (convenience wrapper).

    Args:
        image_path: Path to input image
        use_ai: Enable AI-based corrections
        debug: Save debug information

    Returns:
        Dict or None: Parsed sheet data or None if parsing failed
    """
    gen = parse_images([image_path], use_ai=use_ai, debug=debug)

    # Consume all progress updates
    try:
        while True:
            next(gen)
    except StopIteration as e:
        results = e.value
        return results[0] if results else None
