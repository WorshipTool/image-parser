#!/usr/bin/env python3
"""Debug script pro vizualizaci detekce rotace"""

import cv2
import numpy as np
from pathlib import Path
import sys

def debug_rotation_detection(image_path: str):
    """Vizualizuje každý krok detekce rotace"""

    # Načtení obrázku
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Nelze načíst obrázek: {image_path}")
        return

    # Konverze na grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]
    print(f"📏 Image size: {w}x{h}")

    # KROK 1: Center crop
    margin_w = int(w * 0.15)
    margin_h = int(h * 0.15)
    center_roi = gray[margin_h:h-margin_h, margin_w:w-margin_w]
    print(f"📏 Center ROI size: {center_roi.shape[1]}x{center_roi.shape[0]}")

    # KROK 2: Adaptive threshold
    binary = cv2.adaptiveThreshold(
        center_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # KROK 3: Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print(f"🔍 Found {num_labels-1} components")

    # KROK 4: Filtrování podle plochy
    roi_area = center_roi.shape[0] * center_roi.shape[1]
    min_area = roi_area * 0.0001
    max_area = roi_area * 0.05

    print(f"📊 Area filter: {min_area:.0f} < area < {max_area:.0f}")

    # Vizualizace - vytvoříme barevný obrázek
    vis = cv2.cvtColor(center_roi, cv2.COLOR_GRAY2BGR)

    text_pixels = []
    valid_components = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            valid_components += 1
            mask = (labels == i).astype(np.uint8) * 255
            coords = cv2.findNonZero(mask)
            if coords is not None:
                text_pixels.append(coords.reshape(-1, 2))
                # Vykresli komponentu
                vis[mask > 0] = [0, 255, 0]  # Zelená pro validní komponenty

    print(f"✅ Valid text components: {valid_components}")

    if len(text_pixels) == 0:
        print("❌ No text components found!")
        return

    # KROK 5: minAreaRect
    all_text_pixels = np.vstack(text_pixels)
    rect = cv2.minAreaRect(all_text_pixels)
    ((cx, cy), (width, height), angle) = rect

    print(f"📐 minAreaRect: center=({cx:.1f},{cy:.1f}), size={width:.1f}x{height:.1f}, angle={angle:.2f}°")

    # Vykresli minAreaRect
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)  # Červená

    # Logika pro interpretaci úhlu
    if width < height:
        adjusted_angle = angle - 90
    else:
        adjusted_angle = angle

    # Normalizace
    if adjusted_angle < -45:
        adjusted_angle = adjusted_angle + 90
    elif adjusted_angle > 45:
        adjusted_angle = adjusted_angle - 90

    print(f"📐 Adjusted angle: {adjusted_angle:.2f}°")
    print(f"   (width < height: {width < height})")

    # Alternativní přístup: Hough Lines na binary
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Výpočet úhlu čáry
            line_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Normalizuj na -45 až 45
            if line_angle < -45:
                line_angle += 90
            elif line_angle > 45:
                line_angle -= 90
            if abs(line_angle) < 30:  # Filtruj jen téměř horizontální čáry
                angles.append(line_angle)

        if angles:
            hough_angle = np.median(angles)
            print(f"📐 Hough median angle: {hough_angle:.2f}° (from {len(angles)} lines)")
        else:
            print(f"⚠️  No horizontal lines found in Hough")
    else:
        print(f"⚠️  Hough found no lines")

    # Uložení vizualizace
    output_dir = Path(__file__).parent / "temp"
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(output_dir / "debug_binary.png"), binary)
    cv2.imwrite(str(output_dir / "debug_vis.png"), vis)

    print(f"\n✅ Saved visualizations:")
    print(f"   - temp/debug_binary.png (threshold)")
    print(f"   - temp/debug_vis.png (components + minAreaRect)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_rotation.py <image_path>")
        sys.exit(1)

    debug_rotation_detection(sys.argv[1])
