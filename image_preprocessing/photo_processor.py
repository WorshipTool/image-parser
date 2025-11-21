import cv2
import numpy as np
from typing import Optional
from .utils import four_point_transform, detect_rotation_angle, rotate_image, detect_text_orientation


class PhotoProcessor:
    """
    Zpracovává fotografie - detekuje papír, provádí perspektivní transformaci
    a narovnání.
    """

    def __init__(self, min_area_ratio: float = 0.05, max_area_ratio: float = 0.98):
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

    def _detect_paper(self, image: np.ndarray, debug: bool = True) -> Optional[np.ndarray]:
        """
        Detekuje papír v obrázku pomocí edge detection.

        Args:
            image: Vstupní obrázek
            debug: Pokud True, vypíše debug informace

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

        if debug:
            print(f"      🔍 Trying brightness threshold + edge detection...")

        # METODA 1: Brightness threshold (hledá světlý papír na tmavším pozadí)
        _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours_bright, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_bright) > 0:
            if debug:
                print(f"        Brightness threshold: {len(contours_bright)} bright regions found")

            contours_bright = sorted(contours_bright, key=cv2.contourArea, reverse=True)
            image_area = image.shape[0] * image.shape[1]

            for i, contour in enumerate(contours_bright[:5]):
                area = cv2.contourArea(contour)
                area_ratio = area / image_area

                if debug and i < 3:
                    print(f"          Region {i}: area_ratio={area_ratio:.3f}", end="")

                if self.min_area_ratio < area_ratio < self.max_area_ratio:
                    peri = cv2.arcLength(contour, True)

                    found_valid = False
                    points_history = []
                    best_approx_3_6 = None  # Nejlepší aproximace s 3-6 body
                    # Jemné kroky mezi 0.01 a 0.10
                    epsilons = [0.01, 0.015, 0.02, 0.025, 0.03, 0.032, 0.035, 0.038, 0.04, 0.042, 0.045, 0.048, 0.05, 0.055, 0.06, 0.07, 0.08, 0.10]
                    for epsilon_mult in epsilons:
                        approx = cv2.approxPolyDP(contour, epsilon_mult * peri, True)
                        points_history.append(len(approx))

                        # Ideální: přesně 4 body
                        if len(approx) == 4:
                            if self._is_valid_quadrilateral(approx):
                                score = area_ratio * self._rectangularity_score(approx)

                                if debug and i < 3:
                                    print(f", score={score:.3f} (eps={epsilon_mult}) ✓")

                                if score > best_score:
                                    best_score = score
                                    best_corners = approx.reshape(4, 2)
                                    found_valid = True
                                    break
                            elif debug and i < 3:
                                print(f" (eps={epsilon_mult}: invalid quad)")

                        # Fallback: uložíme nejlepší aproximaci s 3-6 body
                        elif 3 <= len(approx) <= 6 and best_approx_3_6 is None:
                            best_approx_3_6 = approx

                    # Pokud jsme nenašli přesně 4 body, zkusíme aproximaci
                    if not found_valid and best_approx_3_6 is not None:
                        # Vybereme 4 nejvzdálenější rohy
                        points = best_approx_3_6.reshape(-1, 2)
                        if len(points) >= 4:
                            # Najdeme 4 rohy (nejextrémnější body)
                            corners_4 = self._select_4_corners(points)
                            if corners_4 is not None:
                                score = area_ratio * 0.8  # Penalizace za aproximaci
                                if debug and i < 3:
                                    print(f", score={score:.3f} (approx from {len(points)} points) ✓")
                                if score > best_score:
                                    best_score = score
                                    best_corners = corners_4
                                    found_valid = True

                    if not found_valid and debug and i < 3:
                        print(f" (points: {points_history})")

                    if found_valid:
                        break
                elif debug and i < 3:
                    print(f" (rejected: out of range {self.min_area_ratio}-{self.max_area_ratio})")

        # METODA 2: Edge detection (klasická metoda)
        if debug:
            print(f"      🔍 Trying edge detection (Canny)...")

        for threshold1, threshold2 in [(50, 150), (30, 100), (75, 200), (20, 80), (100, 250)]:
            edges = cv2.Canny(blurred, threshold1, threshold2)

            # Dilate a erode pro spojení přerušených hran
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.erode(edges, kernel, iterations=1)

            # Najdeme kontury
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                if debug:
                    print(f"        Threshold ({threshold1},{threshold2}): no contours found")
                continue

            # Seřadíme podle plochy
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            image_area = image.shape[0] * image.shape[1]

            if debug:
                print(f"        Threshold ({threshold1},{threshold2}): {len(contours)} contours, checking top 5")

            # Projdeme největší kontury
            for i, contour in enumerate(contours[:5]):
                area = cv2.contourArea(contour)
                area_ratio = area / image_area

                # Kontrola, zda kontura má rozumnou velikost
                if not (self.min_area_ratio < area_ratio < self.max_area_ratio):
                    if debug and i < 2:
                        print(f"          Contour {i}: area_ratio={area_ratio:.3f} (rejected: out of range)")
                    continue

                # Aproximace kontury na polygon
                peri = cv2.arcLength(contour, True)

                # Zkusíme více approximation tolerancí
                for epsilon_mult in [0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(contour, epsilon_mult * peri, True)

                    # Hledáme čtyřúhelník
                    if len(approx) == 4:
                        # Kontrola, zda je to opravdu čtyřúhelník (ne příliš zkreslený)
                        if self._is_valid_quadrilateral(approx):
                            # Skórování podle toho, jak moc je kontura blízko obdélníku
                            score = area_ratio * self._rectangularity_score(approx)

                            if debug and i < 2:
                                print(f"          Contour {i}: area_ratio={area_ratio:.3f}, score={score:.3f} ✓")

                            if score > best_score:
                                best_score = score
                                best_corners = approx.reshape(4, 2)
                                break  # Našli jsme validní, nemusíme zkoušet další epsilon
                        elif debug and i < 2:
                            print(f"          Contour {i}: area_ratio={area_ratio:.3f} (rejected: invalid quadrilateral)")
                    elif debug and i < 2 and epsilon_mult == 0.02:
                        print(f"          Contour {i}: area_ratio={area_ratio:.3f}, {len(approx)} points (need 4)")

        if debug:
            if best_corners is not None:
                print(f"      ✅ Best quadrilateral found with score={best_score:.3f}")
            else:
                print(f"      ❌ No valid quadrilateral found")

        return best_corners

    def _select_4_corners(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Z více bodů vybere 4 nejvzdálenější rohy (TL, TR, BR, BL).

        Args:
            points: Array bodů (Nx2)

        Returns:
            Array 4 rohů nebo None
        """
        if len(points) < 4:
            return None

        # KROK 1: Odstraníme duplicitní body
        # Použijeme numpy.unique pro odstranění přesně stejných bodů
        points_unique, unique_indices = np.unique(points, axis=0, return_index=True)
        # Seřadíme podle původního pořadí
        points = points_unique[np.argsort(unique_indices)]

        if len(points) < 4:
            return None

        # KROK 2: Najdeme 4 extrémní body
        # Hledáme body, které jsou nejvíc nahoře-vlevo, nahoře-vpravo, dole-vpravo, dole-vlevo

        # Nejlevější bod s minimálním y (top-left)
        tl_idx = np.argmin(points[:, 0] + points[:, 1])
        tl = points[tl_idx]

        # Nejpravější bod s minimálním y (top-right)
        tr_idx = np.argmax(points[:, 0] - points[:, 1])
        tr = points[tr_idx]

        # Nejpravější bod s maximálním y (bottom-right)
        br_idx = np.argmax(points[:, 0] + points[:, 1])
        br = points[br_idx]

        # Nejlevější bod s maximálním y (bottom-left)
        bl_idx = np.argmin(points[:, 0] - points[:, 1])
        bl = points[bl_idx]

        # Zkontrolujeme, že máme 4 různé indexy
        indices = {tl_idx, tr_idx, br_idx, bl_idx}
        if len(indices) < 4:
            # Pokud máme duplicity, zkusíme jiný přístup - vybereme 4 nejvzdálenější body od středu
            center = points.mean(axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            # Seřadíme body podle vzdálenosti od středu
            sorted_indices = np.argsort(distances)[::-1]

            # Vybereme 4 nejvzdálenější body
            if len(sorted_indices) >= 4:
                selected_points = points[sorted_indices[:4]]
                # Seřadíme je jako TL, TR, BR, BL
                sum_pts = selected_points.sum(axis=1)
                diff_pts = np.diff(selected_points, axis=1).flatten()

                tl = selected_points[np.argmin(sum_pts)]
                br = selected_points[np.argmax(sum_pts)]
                tr = selected_points[np.argmax(diff_pts)]
                bl = selected_points[np.argmin(diff_pts)]
            else:
                return None

        # Uspořádáme jako TL, TR, BR, BL
        corners = np.array([tl, tr, br, bl], dtype=np.float32)

        return corners

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

            # Úhel by měl být mezi 20 a 160 stupni (velká tolerance pro perspektivu)
            if angle_deg < 20 or angle_deg > 160:
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
