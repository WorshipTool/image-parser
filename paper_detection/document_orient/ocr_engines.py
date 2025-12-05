"""
OCR engine abstraction for text-based orientation detection
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class OCREngine(ABC):
    """Abstract base class for OCR engines"""

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from image.

        Args:
            image: Input image in BGR format

        Returns:
            Extracted text string
        """
        pass


class TesseractOCR(OCREngine):
    """Tesseract OCR engine implementation"""

    def __init__(self, lang: str = 'eng'):
        """
        Initialize Tesseract OCR.

        Args:
            lang: Language code for OCR (default: 'eng')
        """
        self.lang = lang
        self._tesseract_available = self._check_tesseract()

    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available"""
        try:
            import pytesseract
            # Try to get version to verify it's working
            pytesseract.get_tesseract_version()
            return True
        except (ImportError, Exception):
            return False

    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text using Tesseract OCR.

        Args:
            image: Input image in BGR format

        Returns:
            Extracted text string

        Raises:
            ImportError: If pytesseract is not installed
            RuntimeError: If Tesseract is not properly configured
        """
        if not self._tesseract_available:
            raise ImportError(
                "pytesseract is not available. Install with: pip install pytesseract\n"
                "Also ensure Tesseract OCR is installed on your system:\n"
                "  macOS: brew install tesseract\n"
                "  Ubuntu: apt-get install tesseract-ocr\n"
                "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
            )

        try:
            import pytesseract
            import cv2

            # Convert BGR to RGB for Tesseract
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract text with basic config
            # Use --psm 3 (fully automatic page segmentation) - NO auto-orientation
            # PSM 3 is better at detecting text orientation than PSM 6
            # We still handle rotation ourselves, Tesseract should not rotate
            config = f'--psm 3 -l {self.lang}'
            text = pytesseract.image_to_string(rgb_image, config=config)

            return text.strip()

        except Exception as e:
            # Return empty string on OCR failure rather than crashing
            return ""


class EasyOCR(OCREngine):
    """EasyOCR engine implementation (alternative to Tesseract)"""

    def __init__(self, lang: list = ['en']):
        """
        Initialize EasyOCR.

        Args:
            lang: List of language codes (default: ['en'])
        """
        self.lang = lang
        self._reader = None
        self._easyocr_available = self._check_easyocr()

    def _check_easyocr(self) -> bool:
        """Check if EasyOCR is available"""
        try:
            import easyocr
            return True
        except ImportError:
            return False

    def _get_reader(self):
        """Lazy initialization of EasyOCR reader"""
        if self._reader is None:
            if not self._easyocr_available:
                raise ImportError(
                    "easyocr is not available. Install with: pip install easyocr"
                )
            import easyocr
            self._reader = easyocr.Reader(self.lang, gpu=False)
        return self._reader

    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text using EasyOCR.

        Args:
            image: Input image in BGR format

        Returns:
            Extracted text string

        Raises:
            ImportError: If easyocr is not installed
        """
        try:
            reader = self._get_reader()

            # EasyOCR accepts BGR images directly
            results = reader.readtext(image)

            # Concatenate all detected text
            text = ' '.join([result[1] for result in results])

            return text.strip()

        except Exception as e:
            # Return empty string on OCR failure
            return ""


def create_ocr_engine(engine_name: str = "tesseract", **kwargs) -> OCREngine:
    """
    Factory function to create OCR engine instances.

    Args:
        engine_name: Name of the OCR engine ('tesseract' or 'easyocr')
        **kwargs: Additional arguments passed to the engine constructor

    Returns:
        OCR engine instance

    Raises:
        ValueError: If engine_name is not supported

    Example:
        >>> ocr = create_ocr_engine("tesseract", lang="eng")
        >>> text = ocr.extract_text(image)
    """
    if engine_name.lower() == "tesseract":
        return TesseractOCR(**kwargs)
    elif engine_name.lower() == "easyocr":
        return EasyOCR(**kwargs)
    else:
        raise ValueError(
            f"Unknown OCR engine: {engine_name}. "
            "Supported engines: 'tesseract', 'easyocr'"
        )
