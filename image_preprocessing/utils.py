import cv2
import numpy as np
from enum import Enum
import os
try:
    import pytesseract
except ImportError:
    pytesseract = None


class ImageType(Enum):
    """Typ vstupního obrázku"""
    SCREENSHOT = "screenshot"
    PHOTO = "photo"
    SCAN = "scan"


def detect_image_type(image: np.ndarray, image_path: str = None) -> ImageType:
    """
    Detekuje typ obrázku (screenshot, photo, scan).

    Heuristiky:
    - Screenshot: obvykle má ostrý text, vysoký kontrast, žádné perspektivní zkreslení
    - Photo: může být rozmazaný, má perspektivní zkreslení, různé osvětlení
    - Scan: vysoce kvalitní, rovný, často má černý okraj

    Args:
        image: Vstupní obrázek
        image_path: Cesta k obrázku (použije se pro detekci podle názvu složky)

    Returns:
        Typ obrázku
    """
    # Pokud je cesta k obrázku, zkusíme detekci podle složky
    if image_path:
        path_lower = image_path.lower()
        if 'screenshot' in path_lower:
            return ImageType.SCREENSHOT
        elif 'photo' in path_lower:
            return ImageType.PHOTO
        elif 'scan' in path_lower:
            return ImageType.SCAN

    # Heuristická detekce podle vlastností obrázku
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Výpočet Laplacian variance (míra ostrosti)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Screenshots mají obvykle velmi ostrý text
    if laplacian_var > 1000:
        return ImageType.SCREENSHOT

    # Detekce okrajů pro identifikaci papíru (photos)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pokud najdeme velký čtyřúhelník, pravděpodobně jde o fotku papíru
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        image_area = image.shape[0] * image.shape[1]

        # Pokud největší kontura zabírá 30-95% obrázku, může jít o papír na fotce
        if 0.3 < area / image_area < 0.95:
            peri = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
            if len(approx) == 4:
                return ImageType.PHOTO

    # Výchozí: SCAN
    return ImageType.SCAN


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Seřadí 4 body čtyřúhelníku v pořadí: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array 4 bodů tvaru (4, 2)

    Returns:
        Seřazené body
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left má nejmenší součet, bottom-right největší
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right má nejmenší rozdíl, bottom-left největší
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Provede perspektivní transformaci na základě 4 bodů.

    Args:
        image: Vstupní obrázek
        pts: 4 body definující čtyřúhelník

    Returns:
        Transformovaný obrázek (pohled shora)
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Výpočet šířky výstupního obrázku
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Výpočet výšky výstupního obrázku
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Cílové body pro transformaci
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Výpočet perspektivní transformace
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Otočí obrázek o daný úhel.

    Args:
        image: Vstupní obrázek
        angle: Úhel v stupních (kladný = proti směru hodinových ručiček)

    Returns:
        Otočený obrázek
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Vytvoření rotační matice
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Výpočet nových rozměrů
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Úprava translace
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Rotace
    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))

    return rotated


def detect_rotation_angle(image: np.ndarray) -> float:
    """
    Detekuje úhel rotace textu v obrázku pomocí Hough transform.

    Args:
        image: Vstupní obrázek (grayscale)

    Returns:
        Úhel rotace ve stupních
    """
    # Pokud je obrázek barevný, převedeme na grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detekce hran
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None or len(lines) == 0:
        return 0.0

    # Výpočet průměrného úhlu
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        # Filtrujeme pouze úhly blízko horizontály
        if -45 < angle < 45:
            angles.append(angle)

    if len(angles) == 0:
        return 0.0

    # Medián úhlů (robustnější než průměr)
    median_angle = np.median(angles)
    return median_angle


def detect_text_orientation(image: np.ndarray, debug: bool = False) -> int:
    """
    Detekuje orientaci textu v obrázku pomocí Tesseract OSD.

    Args:
        image: Vstupní obrázek
        debug: Pokud True, vypíše debug informace

    Returns:
        Úhel rotace potřebný k narovnání textu (0, 90, 180, 270)
    """
    # Převod na grayscale pokud je potřeba
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]

    if debug:
        print(f"  🔍 Image dimensions: {w}x{h} (aspect ratio: {w/h:.2f})")

    if pytesseract is None:
        # Fallback: heuristika založená na aspect ratio
        if h > w * 1.3:  # Pokud je výrazně vyšší než širší
            if debug:
                print(f"  ⚠️  No pytesseract, using heuristic: rotate 270° (or -90°)")
            return 270  # Otočit o 270° = -90° = doprava po směru hodinových ručiček
        return 0

    try:
        # Tesseract OSD (Orientation and Script Detection)
        # --psm 0 = pouze OSD, žádné OCR
        use_fallback = False

        try:
            osd = pytesseract.image_to_osd(gray)

            if debug:
                print(f"  📄 Tesseract OSD output:")
                for line in osd.split('\n'):
                    if line.strip():
                        print(f"     {line}")

            # Parse výstupu
            rotation_angle = 0
            orientation_conf = 0
            for line in osd.split('\n'):
                if 'Rotate:' in line:
                    rotation_angle = int(line.split(':')[1].strip())
                if 'Orientation confidence:' in line:
                    orientation_conf = float(line.split(':')[1].strip())

            if debug:
                print(f"  🔄 Detected rotation: {rotation_angle}° (confidence: {orientation_conf:.1f})")

            # Pouze pokud je confidence dostatečně vysoká (min 1.5), použijeme Tesseract výsledek
            if orientation_conf >= 1.5:
                return rotation_angle
            else:
                if debug:
                    print(f"  ⚠️  Confidence too low ({orientation_conf:.1f} < 1.5), using Hough fallback")
                use_fallback = True

        except Exception as e:
            if debug:
                print(f"  ⚠️  Tesseract OSD failed: {e}")
            use_fallback = True

        # Fallback: Použijeme Hough detekci linií k určení orientace
        # (provede se když Tesseract selže NEBO má nízkou confidence)
        if use_fallback:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

            if lines is not None and len(lines) > 5:
                # Spočítáme horizontální vs vertikální linie
                horizontal_count = 0
                vertical_count = 0

                for rho, theta in lines[:, 0]:
                    angle_deg = (theta * 180 / np.pi)

                    # Horizontální linie: kolem 0° nebo 180°
                    if (angle_deg < 20 or angle_deg > 160):
                        horizontal_count += 1
                    # Vertikální linie: kolem 90°
                    elif (70 < angle_deg < 110):
                        vertical_count += 1

                if debug:
                    print(f"  📏 Line detection: {horizontal_count} horizontal, {vertical_count} vertical")

                # Pokud je víc vertikálních než horizontálních, text je otočený o 90°
                # Použijeme nižší threshold (1.2x místo 1.5x) pro lepší detekci
                if vertical_count > horizontal_count * 1.2:
                    if debug:
                        print(f"  📐 Detected vertical text, rotating 270° (or -90°)")
                    return 270

            # Poslední fallback: aspect ratio
            if h > w * 1.3:
                if debug:
                    print(f"  📐 Using aspect ratio heuristic (h>w*1.3): rotate 270° (or -90°)")
                return 270

            if debug:
                print(f"  📐 No clear orientation detected, keeping 0°")
            return 0

    except Exception as e:
        if debug:
            print(f"  ❌ Error: {e}")
        # Fallback: žádná rotace
        return 0
