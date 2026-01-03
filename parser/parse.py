"""
Main parser for extracting sheet components from images

Integrates:
- Paper detection (shouldCrop decision)
- Paper transformation (perspective correction)
- Document orientation (text-based rotation)
- Sheet detection (YOLO-based detection for screenshots)
"""

import warnings
# Suppress all warnings including urllib3 OpenSSL warnings
warnings.filterwarnings('ignore')

import cv2
import os
import sys
from pathlib import Path

# Add current directory to path for submodules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from get_sheet_components import get_sheet_components_from_image
from text_parser import read_and_parse_image


def main():
    """
    Command line interface for parser
    """
    import argparse
    import time

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
        default='temp',
        help='Output directory (default: temp)'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Print debug information'
    )
    parser.add_argument(
        '--ai',
        action='store_true',
        help='Enable AI-based corrections (line_correction and final_smart_correction)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start timing
    start_time = time.time()

    # Track created files for cleanup
    created_files = []

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

        # Save sheet (always 1 image)
        if sheets:
            output_filename = f"{image_path.stem}_sheet.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), sheets[0])
            created_files.append(output_path)

            if args.debug:
                print(f"✓ Saved: {output_path}")

            # Parse text from the sheet
            text_result = read_and_parse_image(sheets[0], debug=args.debug, use_ai=args.ai)

            if text_result:
                # Save parsed text to file
                text_filename = f"{image_path.stem}_sheet.txt"
                text_path = output_dir / text_filename

                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {text_result['title']}\n")
                    f.write("="*60 + "\n\n")
                    f.write(text_result['data'])

                created_files.append(text_path)

                if args.debug:
                    print(f"✓ Saved text: {text_path}")
                    print(f"  Title: {text_result['title']}")

            total_sheets += 1

    # Calculate processing time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Import after text_parser sets up the path
    from ai import get_price
    price = get_price()

    # Summary
    print(f"\n{'='*60}")
    print(f"✓ Processed {len(args.images)} image(s) | {total_sheets} sheet(s)  | {elapsed_time:.2f}s | {price['cost_czk_formatted']}")
    print('='*60)

    # Cleanup created files if not in debug mode
    if not args.debug and created_files:
        for file_path in created_files:
            try:
                if os.path.exists(str(file_path)):
                    os.remove(str(file_path))
            except Exception as e:
                print(f"Warning: Failed to remove {file_path}: {e}")


if __name__ == "__main__":
    main()
