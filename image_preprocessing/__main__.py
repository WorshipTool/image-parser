#!/usr/bin/env python3
"""
CLI interface pro image preprocessing modul.

Použití:
    python -m image_preprocessing -i input.jpg
    python -m image_preprocessing -i input.jpg -o output.png
"""

import argparse
import sys
import os
from pathlib import Path

from .preprocessor import ImagePreprocessor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Image preprocessing pro OCR písní',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Příklady použití:

  # Zpracovat obrázek (vytvoří temp/image_processed.png)
  python -m image_preprocessing -i image.jpg

  # Zpracovat a uložit do konkrétního souboru
  python -m image_preprocessing -i image.jpg -o output.png

Zpracování:
  - Automatická detekce typu (screenshot/photo/scan)
  - Screenshot: YOLO crop na oblast s písní
  - Photo: perspektivní transformace
  - Grayscale + denoising (bez threshold)
  - Výchozí výstup: temp/ složka
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Vstupní obrázek'
    )

    parser.add_argument(
        '-o', '--output',
        help='Výstupní soubor (výchozí: temp/input_processed.png)'
    )

    parser.add_argument(
        '--model',
        default='yolo8best.pt',
        help='Cesta k YOLO modelu (výchozí: yolo8best.pt)'
    )

    return parser.parse_args()


def main():
    """Main CLI function"""
    args = parse_args()

    # Resolve model path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        module_dir = Path(__file__).parent.parent
        model_path = module_dir / args.model

    if not model_path.exists():
        print(f"❌ Chyba: YOLO model nenalezen: {model_path}")
        print(f"   Zkus specifikovat cestu pomocí --model")
        sys.exit(1)

    # Check input exists
    if not Path(args.input).exists():
        print(f"❌ Chyba: Vstupní soubor nenalezen: {args.input}")
        sys.exit(1)

    # Initialize preprocessor
    try:
        preprocessor = ImagePreprocessor(str(model_path))
    except Exception as e:
        print(f"❌ Chyba při inicializaci preprocessoru: {e}")
        sys.exit(1)

    # Process image
    try:
        print(f"📄 Zpracování: {Path(args.input).name}")
        output_path = preprocessor.preprocess(args.input, args.output)
        print(f"✅ Hotovo: {output_path}")
    except Exception as e:
        print(f"❌ Chyba při zpracování: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
