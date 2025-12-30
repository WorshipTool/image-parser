"""
CLI entry point for text_parser module

Allows running: python -m parser.text_parser <image>
"""
import sys
import argparse
from pathlib import Path
import json

from . import read_and_parse_image


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Extract text and chords from sheet music images using OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse single image with debug output
  python -m parser.text_parser sheet.jpg --debug

  # Parse and save to JSON file
  python -m parser.text_parser sheet.jpg -o output.json

  # Parse and save formatted text
  python -m parser.text_parser sheet.jpg -o output.txt --format txt
        """
    )

    parser.add_argument(
        'image',
        type=str,
        help='Input image path'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path (JSON or TXT based on --format)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'txt'],
        default='json',
        help='Output format: json (default) or txt'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Print debug information'
    )

    args = parser.parse_args()

    # Parse image
    result = read_and_parse_image(args.image, debug=args.debug)

    if result is None:
        print(f"✗ Failed to parse image: {args.image}")
        sys.exit(1)

    # Output result
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved JSON to: {output_path}")
        else:  # txt
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Title: {result['title']}\n")
                f.write("="*80 + "\n\n")
                f.write(result['data'])
            print(f"✓ Saved text to: {output_path}")
    else:
        # Print to console
        print("\n" + "="*80)
        print(f"Title: {result['title']}")
        print("="*80)
        print(result['data'])
        print("="*80)


if __name__ == "__main__":
    main()
