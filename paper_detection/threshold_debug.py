"""
Debug s použitím thresholding pro detekci bílých oblastí
"""

import cv2
import numpy as np
from pathlib import Path


def threshold_debug():
    """Debug detekce pomocí thresholding"""

    # Načtení obrázku
    image_path = Path(__file__).parent.parent / "images" / "photos" / "IMG_20230826_093159.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        print("Chyba: Nepodařilo se načíst obrázek")
        return

    print(f"Rozměry obrázku: {image.shape[1]}x{image.shape[0]} px")

    # Převod do šedi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Rozmazání
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    output_dir = Path(__file__).parent / "threshold_debug_output"
    output_dir.mkdir(exist_ok=True)

    # Vyzkoušet různé threshold hodnoty
    threshold_values = [180, 190, 200, 210, 220]

    for thresh_val in threshold_values:
        print(f"\n=== Threshold {thresh_val} ===")

        # Binary threshold - detekovat světlé oblasti
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        # Morfologické operace pro vyčištění
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Nalezení kontur
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        print(f"Nalezeno kontur: {len(contours)}")

        # Minimální plocha
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * 0.01

        # Test různých epsilon
        epsilon_values = [0.01, 0.02, 0.03]

        for epsilon in epsilon_values:
            print(f"  Epsilon: {epsilon}")
            candidates = []

            for contour in contours:
                area = cv2.contourArea(contour)

                if area < min_area:
                    continue

                # Aproximace
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon * peri, True)

                if len(approx) == 4:
                    # Výpočet jasnosti
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [approx.astype(np.int32)], 255)
                    mean_brightness = cv2.mean(gray, mask=mask)[0]

                    # Skóre
                    score = area * (mean_brightness / 255.0)

                    candidates.append({
                        'contour': approx,
                        'area': area,
                        'brightness': mean_brightness,
                        'score': score
                    })

            if candidates:
                # Seřazení
                candidates.sort(key=lambda x: x['score'], reverse=True)

                print(f"    Nalezeno {len(candidates)} čtyřúhelníků:")

                # Ukázat top 3
                for i, cand in enumerate(candidates[:3]):
                    print(f"      {i+1}. Plocha={int(cand['area'])}, "
                          f"Jasnost={cand['brightness']:.1f}, "
                          f"Skóre={int(cand['score'])}")

                    # Vizualizace
                    viz = image.copy()

                    # Overlay
                    overlay = viz.copy()
                    cv2.fillPoly(overlay, [cand['contour'].astype(np.int32)], (255, 200, 100))
                    viz = cv2.addWeighted(overlay, 0.3, viz, 0.7, 0)

                    # Rámeček
                    cv2.polylines(viz, [cand['contour'].astype(np.int32)], True, (255, 100, 0), 3)

                    # Rohy
                    for corner in cand['contour']:
                        cv2.circle(viz, tuple(corner[0]), 10, (255, 0, 0), -1)

                    # Text
                    cv2.putText(
                        viz,
                        f"Threshold: {thresh_val}, Eps: {epsilon}, Rank: {i+1}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )

                    # Uložit
                    output_path = output_dir / f"thresh{thresh_val}_eps{epsilon}_rank{i+1}.jpg"
                    cv2.imwrite(str(output_path), viz)

            else:
                print("    Žádné čtyřúhelníky")

        # Uložit binary
        cv2.imwrite(str(output_dir / f"binary_{thresh_val}.jpg"), binary)

    print(f"\nVýstupy uloženy v: {output_dir}")


if __name__ == "__main__":
    threshold_debug()
