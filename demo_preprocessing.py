#!/usr/bin/env python3
"""
Demo skript pro testování image preprocessing modulu.

Tento skript:
1. Načte vzorové obrázky ze složky images/
2. Aplikuje předzpracování
3. Uloží výsledky do tmp/preprocessing_results/
"""

import os
import cv2
import sys
from pathlib import Path

from image_preprocessing import ImagePreprocessor
from image_preprocessing.utils import ImageType


def main():
    # Cesty
    current_dir = Path(__file__).parent
    images_dir = current_dir / "images"
    model_path = current_dir / "yolo8best.pt"
    output_dir = current_dir / "tmp" / "preprocessing_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Inicializace preprocessoru
    print("Inicializace preprocessoru...")
    preprocessor = ImagePreprocessor(str(model_path))

    # Zpracování screenshotů
    print("\n=== Zpracování screenshotů ===")
    screenshot_dir = images_dir / "screenshots"
    if screenshot_dir.exists():
        screenshots = [f for f in screenshot_dir.iterdir()
                      if f.suffix.lower() in ['.png', '.jpg', '.jpeg'] and not f.name.startswith('.')]

        for i, img_path in enumerate(screenshots[:5]):  # Prvních 5 screenshotů
            print(f"\n{i+1}. Zpracování: {img_path.name}")

            try:
                # Předzpracování (grayscale + threshold)
                processed = preprocessor.preprocess(str(img_path), ImageType.SCREENSHOT)

                # Předzpracování (barevné, bez threshold)
                processed_color = preprocessor.preprocess_keep_color(str(img_path), ImageType.SCREENSHOT)

                # Uložení výsledků
                output_gray = output_dir / f"screenshot_{i+1}_gray.png"
                output_color = output_dir / f"screenshot_{i+1}_color.png"

                cv2.imwrite(str(output_gray), processed)
                cv2.imwrite(str(output_color), processed_color)

                print(f"   ✓ Uloženo: {output_gray.name}, {output_color.name}")

                # Detekce všech písní
                detections = preprocessor.screenshot_processor.detect_all_sheets(str(img_path))
                if detections:
                    print(f"   ✓ Detekovány {len(detections)} oblasti:")
                    for det in detections:
                        print(f"      - {det['class']}: confidence={det['confidence']:.2f}")

            except Exception as e:
                print(f"   ✗ Chyba: {e}")

    # Zpracování fotek
    print("\n=== Zpracování fotek ===")
    photos_dir = images_dir / "photos"
    if photos_dir.exists():
        photos = [f for f in photos_dir.iterdir()
                 if f.suffix.lower() in ['.jpg', '.jpeg'] and not f.name.startswith('.')]

        for i, img_path in enumerate(photos[:5]):  # Prvních 5 fotek
            print(f"\n{i+1}. Zpracování: {img_path.name}")

            try:
                # Předzpracování (grayscale + threshold)
                processed = preprocessor.preprocess(str(img_path), ImageType.PHOTO)

                # Předzpracování (barevné, bez threshold)
                processed_color = preprocessor.preprocess_keep_color(str(img_path), ImageType.PHOTO)

                # Uložení výsledků
                output_gray = output_dir / f"photo_{i+1}_gray.png"
                output_color = output_dir / f"photo_{i+1}_color.png"

                cv2.imwrite(str(output_gray), processed)
                cv2.imwrite(str(output_color), processed_color)

                print(f"   ✓ Uloženo: {output_gray.name}, {output_color.name}")

                # Porovnání původní velikosti a zpracované
                original = cv2.imread(str(img_path))
                print(f"   Původní rozměr: {original.shape[1]}x{original.shape[0]}")
                print(f"   Zpracovaný rozměr: {processed_color.shape[1]}x{processed_color.shape[0]}")

            except Exception as e:
                print(f"   ✗ Chyba: {e}")

    # Zpracování scanů
    print("\n=== Zpracování scanů ===")
    scans_dir = images_dir / "scans"
    if scans_dir.exists():
        scans = [f for f in scans_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and not f.name.startswith('.')]

        for i, img_path in enumerate(scans[:3]):  # První 3 scany
            print(f"\n{i+1}. Zpracování: {img_path.name}")

            try:
                # Předzpracování
                processed = preprocessor.preprocess(str(img_path), ImageType.SCAN)

                # Uložení výsledků
                output_gray = output_dir / f"scan_{i+1}_gray.png"

                cv2.imwrite(str(output_gray), processed)

                print(f"   ✓ Uloženo: {output_gray.name}")

            except Exception as e:
                print(f"   ✗ Chyba: {e}")

    print(f"\n✓ Hotovo! Výsledky uloženy v: {output_dir}")
    print(f"\nPočet zpracovaných obrázků: {len(list(output_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
