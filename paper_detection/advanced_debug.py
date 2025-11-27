"""
Pokročilý debug skript s vizualizací kandidátů a jejich skóre
"""

import cv2
import numpy as np
from pathlib import Path


def advanced_debug():
    """Debug s ukázkou všech kandidátů"""

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

    # Test různých prahů
    threshold_configs = [
        (30, 100),
        (50, 150),
    ]

    epsilon_values = [0.01, 0.015, 0.02, 0.03]

    output_dir = Path(__file__).parent / "advanced_debug_output"
    output_dir.mkdir(exist_ok=True)

    for thresh1, thresh2 in threshold_configs:
        print(f"\n=== Canny({thresh1}, {thresh2}) ===")

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

        print(f"Nalezeno kontur: {len(contours)}")

        # Minimální plocha (1% - papír může být menší)
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * 0.01

        for epsilon in epsilon_values:
            print(f"\n  Epsilon: {epsilon}")
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

                    # Filtr na minimální jasnost (papír je světlý)
                    if mean_brightness < 150:
                        continue

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

                # Ukázat top 5
                for i, cand in enumerate(candidates[:5]):
                    print(f"      {i+1}. Plocha={int(cand['area'])}, "
                          f"Jasnost={cand['brightness']:.1f}, "
                          f"Skóre={int(cand['score'])}")

                    # Vizualizace
                    viz = image.copy()

                    # Nakreslit overlay
                    overlay = viz.copy()
                    cv2.fillPoly(overlay, [cand['contour'].astype(np.int32)], (255, 200, 100))
                    viz = cv2.addWeighted(overlay, 0.3, viz, 0.7, 0)

                    # Nakreslit rámeček
                    cv2.polylines(viz, [cand['contour'].astype(np.int32)], True, (255, 100, 0), 3)

                    # Nakreslit rohy
                    for corner in cand['contour']:
                        cv2.circle(viz, tuple(corner[0]), 10, (255, 0, 0), -1)

                    # Přidat text s info
                    cv2.putText(
                        viz,
                        f"Rank: {i+1}, Score: {int(cand['score'])}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        viz,
                        f"Area: {int(cand['area'])}, Brightness: {cand['brightness']:.1f}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )

                    # Uložit
                    output_path = output_dir / f"candidate_t{thresh1}_{thresh2}_e{epsilon}_rank{i+1}.jpg"
                    cv2.imwrite(str(output_path), viz)

            else:
                print("    Žádné čtyřúhelníky")

    print(f"\nVýstupy uloženy v: {output_dir}")


if __name__ == "__main__":
    advanced_debug()
