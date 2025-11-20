import cv2
import numpy as np
from typing import Optional
from .utils import four_point_transform, detect_rotation_angle, rotate_image, detect_text_orientation


class PhotoProcessor:
    """
    Zpracovává fotografie - detekuje papír, provádí perspektivní transformaci
    a narovnání.
    """

    def __init__(self, min_area_ratio: float = 0.3, max_area_ratio: float = 0.95):
        """
        Inicializace procesoru pro fotografie.

        Args:
            min_area_ratio: Minimální poměr plochy papíru k celkové ploše obrázku
            max_area_ratio: Maximální poměr plochy papíru k celkové ploše obrázku
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Zpracuje fotografii - najde papír a provede perspektivní transformaci.

        Args:
            image: Vstupní obrázek

        Returns:
            Transformovaný a narovnaný obrázek
        """
        # Detekce papíru
        corners = self._detect_paper(image)

        if corners is not None:
            # Perspektivní transformace
            warped = four_point_transform(image, corners)
        else:
            # Pokud se papír nenajde, použijeme originální obrázek
            warped = image

        # Nejdříve detekce a korekce orientace textu (90°, 180°, 270°)
        orientation_angle = detect_text_orientation(warped, debug=False)
        if orientation_angle != 0:
            warped = rotate_image(warped, -orientation_angle)

        # Pak detekce a korekce drobné rotace (do 45°)
        angle = detect_rotation_angle(warped)

        # Pouze pokud je úhel výrazný (více než 0.5 stupně)
        if abs(angle) > 0.5:
            warped = rotate_image(warped, angle)

        return warped

    def _detect_paper(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detekuje papír v obrázku pomocí edge detection.

        Args:
            image: Vstupní obrázek

        Returns:
            Array 4 rohů papíru nebo None
        """
        # Převod na grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian blur pro redukci šumu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection - zkusíme několik prahů
        best_corners = None
        best_score = 0

        for threshold1, threshold2 in [(50, 150), (30, 100), (75, 200)]:
            edges = cv2.Canny(blurred, threshold1, threshold2)

            # Dilate a erode pro spojení přerušených hran
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.erode(edges, kernel, iterations=1)

            # Najdeme kontury
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            # Seřadíme podle plochy
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            image_area = image.shape[0] * image.shape[1]

            # Projdeme největší kontury
            for contour in contours[:5]:
                area = cv2.contourArea(contour)
                area_ratio = area / image_area

                # Kontrola, zda kontura má rozumnou velikost
                if not (self.min_area_ratio < area_ratio < self.max_area_ratio):
                    continue

                # Aproximace kontury na polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # Hledáme čtyřúhelník
                if len(approx) == 4:
                    # Kontrola, zda je to opravdu čtyřúhelník (ne příliš zkreslený)
                    if self._is_valid_quadrilateral(approx):
                        # Skórování podle toho, jak moc je kontura blízko obdélníku
                        score = area_ratio * self._rectangularity_score(approx)

                        if score > best_score:
                            best_score = score
                            best_corners = approx.reshape(4, 2)

        return best_corners

    def _is_valid_quadrilateral(self, approx: np.ndarray) -> bool:
        """
        Kontroluje, zda je čtyřúhelník validní (ne příliš zkreslený).

        Args:
            approx: Aproximovaná kontura s 4 body

        Returns:
            True pokud je validní
        """
        # Kontrola úhlů - žádný úhel by neměl být příliš ostrý
        points = approx.reshape(4, 2)

        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]

            # Vypočítáme úhel
            v1 = p1 - p2
            v2 = p3 - p2

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = np.degrees(angle)

            # Úhel by měl být mezi 45 a 135 stupni (tolerance pro perspektivu)
            if angle_deg < 45 or angle_deg > 135:
                return False

        return True

    def _rectangularity_score(self, approx: np.ndarray) -> float:
        """
        Vypočítá skóre "obdélníkovosti" čtyřúhelníku (0-1).

        Args:
            approx: Aproximovaná kontura s 4 body

        Returns:
            Skóre 0-1, kde 1 = perfektní obdélník
        """
        points = approx.reshape(4, 2)

        # Vypočítáme délky všech 4 stran
        side_lengths = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            length = np.linalg.norm(p2 - p1)
            side_lengths.append(length)

        # Ideální obdélník má protilehlé strany stejně dlouhé
        ratio1 = min(side_lengths[0], side_lengths[2]) / (max(side_lengths[0], side_lengths[2]) + 1e-6)
        ratio2 = min(side_lengths[1], side_lengths[3]) / (max(side_lengths[1], side_lengths[3]) + 1e-6)

        # Průměrné skóre
        score = (ratio1 + ratio2) / 2.0

        return score
