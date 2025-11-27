"""
Debug skript pro ladění detekce papíru
"""

import cv2
import numpy as np
from pathlib import Path


def debug_detection():
    """Debug detekce s vizualizací mezikroků"""

    # Načtení obrázku
    image_path = Path(__file__).parent.parent / "images" / "photos" / "IMG_20230826_093159.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        print("Chyba: Nepodařilo se načíst obrázek")
        return

    print(f"Rozměry obrázku: {image.shape[1]}x{image.shape[0]} px")

    # Převod do šedi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("1. Převod do šedi - OK")

    # Rozmazání
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("2. Gaussian blur - OK")

    # Vyzkoušet různé prahy pro Canny
    thresholds = [
        (30, 100),
        (50, 150),
        (75, 200),
        (100, 250)
    ]

    output_dir = Path(__file__).parent / "debug_output"
    output_dir.mkdir(exist_ok=True)

    for i, (thresh1, thresh2) in enumerate(thresholds):
        print(f"\n=== Test {i+1}: Canny({thresh1}, {thresh2}) ===")

        # Detekce hran
        edges = cv2.Canny(blurred, thresh1, thresh2)

        # Dilatace
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Nalezení kontur
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        print(f"  Nalezeno kontur: {len(contours)}")

        if contours:
            # Seřazení podle plochy
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            for j, contour in enumerate(sorted_contours):
                area = cv2.contourArea(contour)
                peri = cv2.arcLength(contour, True)

                # Aproximace
                for epsilon in [0.01, 0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(contour, epsilon * peri, True)

                    if len(approx) == 4:
                        print(f"  Kontura {j+1}: Plocha={int(area)}, Obvod={int(peri)}, Epsilon={epsilon} -> 4 rohy!")

                        # Vizualizace
                        viz = image.copy()
                        cv2.drawContours(viz, [approx], -1, (0, 255, 0), 3)
                        for corner in approx:
                            cv2.circle(viz, tuple(corner[0]), 10, (255, 0, 0), -1)

                        # Uložení
                        output_path = output_dir / f"debug_thresh{thresh1}_{thresh2}_contour{j+1}_eps{epsilon}.jpg"
                        cv2.imwrite(str(output_path), viz)
                        print(f"    Uloženo: {output_path.name}")
                        break

        # Uložení edges a dilated
        cv2.imwrite(str(output_dir / f"edges_{thresh1}_{thresh2}.jpg"), edges)
        cv2.imwrite(str(output_dir / f"dilated_{thresh1}_{thresh2}.jpg"), dilated)

    print(f"\nDebug obrázky uloženy v: {output_dir}")


if __name__ == "__main__":
    debug_detection()
