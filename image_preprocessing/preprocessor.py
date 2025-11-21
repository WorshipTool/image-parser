import cv2
import numpy as np
from typing import Optional, Tuple
from .screenshot_processor import ScreenshotProcessor
from .photo_processor import PhotoProcessor
from .utils import detect_text_orientation, detect_rotation_angle, rotate_image, four_point_transform
try:
    import pytesseract
except ImportError:
    pytesseract = None


class ImagePreprocessor:
    """
    Hlavní třída pro předzpracování obrázků písní.

    Univerzální přístup:
    1. Pokusí se najít oblast s písní (YOLO) nebo papír (edge detection)
    2. Ořízne/vyřízne nalezenou oblast
    3. Aplikuje perspektivní transformaci pokud je detekován papír
    4. Otočí text do správné orientace (horizontálně)
    5. Opraví drobnou rotaci
    6. Aplikuje grayscale + denoising
    """

    def __init__(self, yolo_model_path: str):
        """
        Inicializace preprocessoru.

        Args:
            yolo_model_path: Cesta k YOLO modelu pro detekci písní
        """
        self.screenshot_processor = ScreenshotProcessor(yolo_model_path)
        self.photo_processor = PhotoProcessor()

    def preprocess(self, image_path: str, output_path: Optional[str] = None) -> list:
        """
        Předzpracuje obrázek - univerzální logika pro všechny typy.

        Najde všechny písně na obrázku a zpracuje každou samostatně.

        Kroky:
        1. YOLO najde všechny písně na obrázku
        2. Pro každou píseň:
           a) Cropne oblast
           b) Zkusí paper detection v okolí
           c) Aplikuje perspektivní transformaci pokud najde papír
           d) Otočí text do správné orientace
           e) Opraví rotaci
           f) Aplikuje grayscale + denoising
           g) Uloží jako samostatný soubor

        Args:
            image_path: Cesta k vstupnímu obrázku
            output_path: Cesta k výstupnímu souboru (volitelné, pro jednu píseň)

        Returns:
            List cest k výstupním souborům
        """
        # Načtení obrázku
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Nepodařilo se načíst obrázek: {image_path}")

        # KROK 1: YOLO detekce všech písní
        detected_boxes = self.screenshot_processor.detect_all_songs(image_path, image)

        output_paths = []

        if len(detected_boxes) == 0:
            # YOLO nenašlo nic - zkusíme paper detection na celém obrázku
            print("  ⚠️  No songs detected by YOLO, trying paper detection...")
            paper_corners = self.photo_processor._detect_paper(image, debug=False)

            if paper_corners is not None:
                processed = four_point_transform(image, paper_corners)
            else:
                processed = image

            # Zpracuj jako jednu píseň
            processed = self._process_single_song(processed)

            # Ulož
            final_output_path = self._save_processed(image_path, processed, output_path, song_index=None)
            output_paths.append(final_output_path)

        else:
            # Zpracuj každou detekovanou píseň
            print(f"  ✅ Found {len(detected_boxes)} song(s)")

            for i, (x1, y1, x2, y2) in enumerate(detected_boxes):
                print(f"  📝 Processing song {i+1}/{len(detected_boxes)}...")

                # Přidáme padding
                h, w = image.shape[:2]
                padding_x = int((x2 - x1) * 0.05)
                padding_y = int((y2 - y1) * 0.05)

                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(w, x2 + padding_x)
                y2 = min(h, y2 + padding_y)

                # Crop
                cropped = image[y1:y2, x1:x2]

                # Zkusíme paper detection v cropnuté oblasti
                paper_corners = self.photo_processor._detect_paper(cropped, debug=False)

                if paper_corners is not None:
                    print(f"     📄 Paper detected, applying perspective transform")
                    processed = four_point_transform(cropped, paper_corners)
                else:
                    processed = cropped

                # Zpracuj píseň
                processed = self._process_single_song(processed)

                # Ulož
                song_output = output_path if len(detected_boxes) == 1 else None
                final_output_path = self._save_processed(image_path, processed, song_output, song_index=i+1)
                output_paths.append(final_output_path)

        return output_paths

    def _process_single_song(self, image: np.ndarray) -> np.ndarray:
        """
        Zpracuje jednu píseň: rotace + denoising.

        Args:
            image: Vstupní obrázek

        Returns:
            Zpracovaný obrázek
        """
        # KROK 1: Otočit do správné orientace (90°/180°/270°)
        orientation_angle = detect_text_orientation(image, debug=False)
        if orientation_angle != 0:
            image = rotate_image(image, -orientation_angle)

        # KROK 2: Opravit rotaci (Hough)
        angle = detect_rotation_angle(image)

        # Pokud je úhel větší než 15°, zkusíme všechny orientace a vybereme nejlepší
        if abs(angle) > 15:
            # Zkusíme 4 orientace a vybereme tu s nejmenším zbytkovým úhlem
            candidates = []

            # 0° (žádná rotace)
            angle_0 = abs(detect_rotation_angle(image))
            candidates.append((0, angle_0, image))

            # 90° doleva
            rotated_90 = rotate_image(image, 90)
            angle_90 = abs(detect_rotation_angle(rotated_90))
            candidates.append((90, angle_90, rotated_90))

            # 90° doprava (-90°)
            rotated_m90 = rotate_image(image, -90)
            angle_m90 = abs(detect_rotation_angle(rotated_m90))
            candidates.append((-90, angle_m90, rotated_m90))

            # 180°
            rotated_180 = rotate_image(image, 180)
            angle_180 = abs(detect_rotation_angle(rotated_180))
            candidates.append((180, angle_180, rotated_180))

            # Vyber orientaci s nejmenším zbytkovým úhlem
            best_rotation, best_angle, best_image = min(candidates, key=lambda x: x[1])

            # Pokud je i nejlepší residual angle > 10°, Hough není spolehlivý
            # Zkusíme OCR-based metodu
            if best_angle > 10 and pytesseract is not None:
                ocr_candidates = []
                for rotation, _, img in candidates:
                    score = self._ocr_text_score(img)
                    ocr_candidates.append((rotation, score, img))

                # Vyber orientaci s nejvyšším OCR score
                best_rotation, best_score, best_image = max(ocr_candidates, key=lambda x: x[1])
                image = best_image
                angle = detect_rotation_angle(image)
            else:
                image = best_image
                angle = detect_rotation_angle(image)

        # Pak aplikuj jemnou rotaci
        if abs(angle) > 0.5:
            image = rotate_image(image, angle)

        # KROK 3: Základní předzpracování
        image = self._basic_preprocessing(image)

        return image

    def _save_processed(self, image_path: str, processed: np.ndarray, output_path: Optional[str], song_index: Optional[int]) -> str:
        """
        Uloží zpracovaný obrázek.

        Args:
            image_path: Cesta k originálnímu obrázku
            processed: Zpracovaný obrázek
            output_path: Volitelná výstupní cesta
            song_index: Index písně (pokud jich je více)

        Returns:
            Cesta k uloženému souboru
        """
        if output_path is None:
            from pathlib import Path
            input_file = Path(image_path)
            temp_dir = Path(__file__).parent.parent / "temp"
            temp_dir.mkdir(exist_ok=True)

            if song_index is None:
                output_path = str(temp_dir / f"{input_file.stem}_processed.png")
            else:
                output_path = str(temp_dir / f"{input_file.stem}_song{song_index}_processed.png")

        cv2.imwrite(output_path, processed)
        return output_path

    def _ocr_text_score(self, image: np.ndarray) -> float:
        """
        Spustí rychlé OCR a vrátí skóre (počet detekovaných znaků).
        Vyšší skóre = více textu nalezeno = pravděpodobně správná orientace.

        Args:
            image: Vstupní obrázek

        Returns:
            Skóre (počet alfanumerických znaků)
        """
        if pytesseract is None:
            return 0.0

        try:
            # Převod na grayscale pokud je potřeba
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Aplikuj denoising pro lepší OCR
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

            # Resize pokud je obrázek příliš malý (OCR funguje lépe na větších obrázcích)
            h, w = denoised.shape
            if h < 300:
                scale = 300 / h
                denoised = cv2.resize(denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Rychlé OCR s minimální konfigurací
            text = pytesseract.image_to_string(denoised, config='--psm 6 --oem 1')

            # Spočítej alfanumerické znaky (ignoruj whitespace)
            alnum_count = sum(c.isalnum() for c in text)

            return float(alnum_count)

        except Exception:
            return 0.0

    def _basic_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Základní předzpracování - vždy stejné:
        1. Grayscale
        2. Denoising
        3. BEZ threshold (zachová odstíny šedi)

        Args:
            image: Vstupní obrázek

        Returns:
            Předzpracovaný obrázek (grayscale s denoisingem)
        """
        # 1. Převod na grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 2. Odstranění šumu
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # 3. BEZ threshold - vrátíme grayscale s denoisingem
        return denoised
