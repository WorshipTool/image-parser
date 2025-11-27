import cv2
import numpy as np
from typing import Optional, List
from pathlib import Path
from ultralytics import YOLO
from .utils import detect_text_orientation, detect_rotation_angle, rotate_image


class ImagePreprocessor:
    """
    Hlavní třída pro předzpracování obrázků písní.

    Zjednodušený univerzální přístup:
    1. YOLO najde všechny písně na obrázku
    2. Pro každou píseň:
       - Crop s paddingem
       - Detekce orientace + rotace (Tesseract OSD + Hough fallback)
       - Hough rotační korekce
       - Grayscale + denoising
    """

    def __init__(self, yolo_model_path: str):
        """
        Inicializace preprocessoru.

        Args:
            yolo_model_path: Cesta k YOLO modelu pro detekci písní
        """
        self.model = YOLO(yolo_model_path)

    def preprocess(self, image_path: str, output_path: Optional[str] = None) -> List[str]:
        """
        Předzpracuje obrázek - najde všechny písně a zpracuje je.

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

        # YOLO detekce všech písní
        detected_boxes = self._detect_all_songs(image_path, image)
        output_paths = []

        if len(detected_boxes) == 0:
            # YOLO nenašlo nic - zpracuj celý obrázek
            print("  ⚠️  No songs detected by YOLO, processing whole image...")
            processed = self._process_single_song(image)
            final_output_path = self._save_processed(image_path, processed, output_path, song_index=None)
            output_paths.append(final_output_path)
        else:
            # Zpracuj každou detekovanou píseň
            print(f"  ✅ Found {len(detected_boxes)} song(s)")

            for i, (x1, y1, x2, y2) in enumerate(detected_boxes):
                print(f"  📝 Processing song {i+1}/{len(detected_boxes)}...")

                # YOLO crop s paddingem (5%)
                h, w = image.shape[:2]
                padding_x = int((x2 - x1) * 0.05)
                padding_y = int((y2 - y1) * 0.05)

                x1_crop = max(0, x1 - padding_x)
                y1_crop = max(0, y1 - padding_y)
                x2_crop = min(w, x2 + padding_x)
                y2_crop = min(h, y2 + padding_y)

                cropped = image[y1_crop:y2_crop, x1_crop:x2_crop]

                # Zpracuj píseň (rotace + denoising)
                processed = self._process_single_song(cropped)

                # Ulož
                song_output = output_path if len(detected_boxes) == 1 else None
                final_output_path = self._save_processed(image_path, processed, song_output, song_index=i+1)
                output_paths.append(final_output_path)

        return output_paths

    def _detect_all_songs(self, image_path: str, image: np.ndarray) -> List[tuple]:
        """
        Detekuje všechny písně na obrázku pomocí YOLO.

        Args:
            image_path: Cesta k obrázku (pro YOLO)
            image: Načtený obrázek (pro kontrolu rozměrů)

        Returns:
            List bounding boxů [(x1, y1, x2, y2), ...]
        """
        results = self.model.predict(image_path, conf=0.25, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        detected_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detected_boxes.append((int(x1), int(y1), int(x2), int(y2)))

        # Seřaď podle velikosti (největší první)
        detected_boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)

        return detected_boxes

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

        # KROK 2: Opravit rotaci (text-based detection)
        angle = detect_rotation_angle(image, debug=True)

        # Jemná rotace (neguj úhel, protože rotace je ve směru hodinových ručiček)
        if abs(angle) > 0.5:
            image = rotate_image(image, -angle)

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
            input_file = Path(image_path)
            temp_dir = Path(__file__).parent.parent / "temp"
            temp_dir.mkdir(exist_ok=True)

            if song_index is None:
                output_path = str(temp_dir / f"{input_file.stem}_processed.png")
            else:
                output_path = str(temp_dir / f"{input_file.stem}_song{song_index}_processed.png")

        cv2.imwrite(output_path, processed)
        return output_path

    def _basic_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Základní předzpracování:
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
