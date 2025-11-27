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


def detect_rotation_angle(image: np.ndarray, debug: bool = False) -> float:
    """
    Detekuje úhel rotace textu v obrázku pomocí analýzy textových komponent.

    Klíčová změna: Počítá úhel z TEXTU (malých komponent), ne z celého obrazu/pozadí.
    Tímto se vyhne detekci hran stolu, kachliček, prken apod.

    Args:
        image: Vstupní obrázek (grayscale)
        debug: Pokud True, vypíše debug informace

    Returns:
        Úhel rotace ve stupních
    """
    # Pokud je obrázek barevný, převedeme na grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]

    # Použij celý obrázek (YOLO už udělal crop)
    center_roi = gray

    # KROK 2: Vytvoř masku textu pomocí adaptive threshold
    # Invertuj, aby text byl bílý (255), pozadí černé (0)
    binary = cv2.adaptiveThreshold(
        center_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # KROK 3: Najdi textové komponenty (connected components)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:  # Pouze pozadí
        return 0.0

    # KROK 4: Filtruj komponenty podle plochy
    # Zahoď obrovské bloby (pozadí, dlouhé čáry) a malé šumy
    # Necháme jen střední komponenty (pravděpodobně písmena)
    roi_area = center_roi.shape[0] * center_roi.shape[1]
    min_area = roi_area * 0.0001  # 0.01% plochy ROI
    max_area = roi_area * 0.05    # 5% plochy ROI (velké čáry/pozadí)

    text_pixels = []
    for i in range(1, num_labels):  # Skip label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            # Přidej všechny pixely této komponenty
            mask = (labels == i).astype(np.uint8) * 255
            coords = cv2.findNonZero(mask)
            if coords is not None:
                text_pixels.append(coords.reshape(-1, 2))

    if len(text_pixels) == 0:
        if debug:
            print(f"  ⚠️  No text components found for rotation detection")
        return 0.0

    # Spojíme všechny textové pixely dohromady
    all_text_pixels = np.vstack(text_pixels)

    # KROK 5: Použij PCA (Principal Component Analysis) pro detekci hlavního směru textu
    # PCA najde hlavní směr rozložení textových pixelů
    try:
        # Vlastní implementace PCA pomocí numpy
        # Centrum dat
        mean = np.mean(all_text_pixels, axis=0)
        centered = all_text_pixels - mean

        # Kovarianční matice
        cov_matrix = np.cov(centered.T)

        # Vlastní čísla a vlastní vektory
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Seřaď podle velikosti vlastních čísel (sestupně)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # První vlastní vektor (hlavní komponenta)
        eigen_vec = eigenvectors[:, 0]

        # Výpočet úhlu z eigenvektoru
        angle = np.degrees(np.arctan2(eigen_vec[1], eigen_vec[0]))

        # Normalizuj na rozsah -45 až +45
        if angle < -45:
            angle = angle + 90
        elif angle > 45:
            angle = angle - 90

        if debug:
            print(f"  📐 Detected rotation angle from text: {angle:.2f}° (PCA)")

        return angle

    except Exception as e:
        if debug:
            print(f"  ⚠️  PCA failed: {e}, using fallback")

    # Fallback na minAreaRect (pokud PCA selže)
    rect = cv2.minAreaRect(all_text_pixels)
    ((cx, cy), (width, height), angle) = rect

    if width < height:
        angle = angle - 90

    if angle < -45:
        angle = angle + 90
    elif angle > 45:
        angle = angle - 90

    if debug:
        print(f"  📐 Detected rotation angle from text: {angle:.2f}° (minAreaRect fallback)")

    return angle


def _check_180_rotation(gray: np.ndarray, debug: bool = False) -> bool:
    """
    Zkontroluje, zda je text otočený o 180° pomocí OCR testu.

    Args:
        gray: Grayscale obrázek
        debug: Pokud True, vypíše debug informace

    Returns:
        True pokud je text otočený o 180°, False jinak
    """
    if pytesseract is None:
        return False

    try:
        # Crop do středu obrázku (pro rychlejší OCR)
        h, w = gray.shape[:2]
        crop_h = min(h // 2, 400)
        crop_w = min(w // 2, 400)
        y_start = (h - crop_h) // 2
        x_start = (w - crop_w) // 2
        cropped = gray[y_start:y_start+crop_h, x_start:x_start+crop_w]

        # OCR na aktuální orientaci
        data_0 = pytesseract.image_to_data(cropped, output_type=pytesseract.Output.DICT, lang='ces')
        conf_0 = [float(c) for c in data_0['conf'] if c != '-1']
        avg_conf_0 = sum(conf_0) / len(conf_0) if conf_0 else 0

        # OCR na rotované o 180°
        rotated = cv2.rotate(cropped, cv2.ROTATE_180)
        data_180 = pytesseract.image_to_data(rotated, output_type=pytesseract.Output.DICT, lang='ces')
        conf_180 = [float(c) for c in data_180['conf'] if c != '-1']
        avg_conf_180 = sum(conf_180) / len(conf_180) if conf_180 else 0

        if debug:
            print(f"  🔄 180° check: current={avg_conf_0:.1f}, rotated={avg_conf_180:.1f}")

        # Pokud je rotované výrazně lepší (20% rozdíl), text je otočený o 180°
        return avg_conf_180 > avg_conf_0 * 1.2

    except Exception as e:
        if debug:
            print(f"  ⚠️  180° check failed: {e}")
        return False


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
                # Speciální případ: Pokud Tesseract řekl 0°, zkontroluj 180°
                # (protože Tesseract někdy nedokáže rozlišit 0° od 180°)
                if rotation_angle == 0:
                    if _check_180_rotation(gray, debug):
                        if debug:
                            print(f"  🔄 OCR confidence better at 180°, overriding Tesseract")
                        return 180
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

            # Finální fallback: 180° OCR test
            if _check_180_rotation(gray, debug):
                if debug:
                    print(f"  🔄 OCR confidence better at 180°, rotating")
                return 180

            if debug:
                print(f"  📐 No clear orientation detected, keeping 0°")
            return 0

    except Exception as e:
        if debug:
            print(f"  ❌ Error: {e}")
        # Fallback: žádná rotace
        return 0
