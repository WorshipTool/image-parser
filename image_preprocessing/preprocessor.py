import cv2
import numpy as np
from typing import Optional, Tuple
from .screenshot_processor import ScreenshotProcessor
from .photo_processor import PhotoProcessor
from .utils import ImageType, detect_image_type


class ImagePreprocessor:
    """
    Hlavní třída pro předzpracování obrázků písní.

    Podporuje dva typy vstupů:
    - Screenshots: používá YOLO model pro detekci oblasti s písní
    - Photos: detekuje papír a provádí perspektivní transformaci
    """

    def __init__(self, yolo_model_path: str):
        """
        Inicializace preprocessoru.

        Args:
            yolo_model_path: Cesta k YOLO modelu pro detekci písní
        """
        self.screenshot_processor = ScreenshotProcessor(yolo_model_path)
        self.photo_processor = PhotoProcessor()

    def preprocess(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Předzpracuje obrázek - vždy stejným způsobem.

        Kroky:
        1. Detekce typu (screenshot/photo/scan)
        2. Screenshot: YOLO crop
        3. Photo: perspektivní transformace
        4. Grayscale
        5. Denoising
        6. Bez threshold (zachová odstíny šedi)

        Args:
            image_path: Cesta k vstupnímu obrázku
            output_path: Cesta k výstupnímu souboru (volitelné)

        Returns:
            Cesta k výstupnímu souboru
        """
        # Načtení obrázku
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Nepodařilo se načíst obrázek: {image_path}")

        # Automatická detekce typu
        image_type = detect_image_type(image, image_path)

        # Zpracování podle typu
        if image_type == ImageType.SCREENSHOT:
            processed = self.screenshot_processor.process(image_path, image)
        elif image_type == ImageType.PHOTO:
            processed = self.photo_processor.process(image)
        else:
            processed = image

        # Základní předzpracování (vždy stejné)
        processed = self._basic_preprocessing(processed)

        # Uložení
        if output_path is None:
            # Výchozí: do temp/ složky s "_processed" příponou
            from pathlib import Path
            input_file = Path(image_path)

            # Vytvoř temp složku, pokud neexistuje
            temp_dir = Path(__file__).parent.parent / "temp"
            temp_dir.mkdir(exist_ok=True)

            output_path = str(temp_dir / f"{input_file.stem}_processed.png")

        cv2.imwrite(output_path, processed)
        return output_path

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

    def preprocess_keep_color(self, image_path: str, image_type: Optional[ImageType] = None) -> np.ndarray:
        """
        Předzpracuje obrázek, ale zachová barevnou verzi (bez grayscale a threshold).
        Užitečné pro vizuální kontrolu nebo další zpracování.

        Args:
            image_path: Cesta k obrázku
            image_type: Typ obrázku (screenshot/photo)

        Returns:
            Předzpracovaný barevný obrázek
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Nepodařilo se načíst obrázek: {image_path}")

        if image_type is None:
            image_type = detect_image_type(image, image_path)

        if image_type == ImageType.SCREENSHOT:
            processed = self.screenshot_processor.process(image_path, image)
        elif image_type == ImageType.PHOTO:
            processed = self.photo_processor.process(image)
        else:
            processed = image

        return processed
