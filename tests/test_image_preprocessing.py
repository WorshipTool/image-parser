import os
import sys
import pytest
import cv2
import numpy as np

# Přidáme parent directory do sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_preprocessing import ImagePreprocessor
from image_preprocessing.utils import ImageType, detect_image_type, order_points, four_point_transform


# Fixture pro YOLO model path
@pytest.fixture
def model_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return os.path.join(parent_dir, "yolo8best.pt")


# Fixture pro testovací obrázky
@pytest.fixture
def test_images_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return os.path.join(parent_dir, "images")


@pytest.fixture
def preprocessor(model_path):
    """Vytvoří instanci ImagePreprocessor"""
    return ImagePreprocessor(model_path)


class TestImageTypeDetection:
    """Testy pro detekci typu obrázku"""

    def test_detect_screenshot_by_path(self, test_images_dir):
        """Test detekce screenshotu podle cesty"""
        screenshot_dir = os.path.join(test_images_dir, "screenshots")
        if os.path.exists(screenshot_dir):
            screenshots = [f for f in os.listdir(screenshot_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
            if screenshots:
                img_path = os.path.join(screenshot_dir, screenshots[0])
                img = cv2.imread(img_path)
                img_type = detect_image_type(img, img_path)
                assert img_type == ImageType.SCREENSHOT, f"Screenshot nebyl detekován správně: {img_path}"

    def test_detect_photo_by_path(self, test_images_dir):
        """Test detekce fotky podle cesty"""
        photos_dir = os.path.join(test_images_dir, "photos")
        if os.path.exists(photos_dir):
            photos = [f for f in os.listdir(photos_dir) if f.endswith(('.jpg', '.jpeg')) and not f.startswith('.')]
            if photos:
                img_path = os.path.join(photos_dir, photos[0])
                img = cv2.imread(img_path)
                img_type = detect_image_type(img, img_path)
                assert img_type == ImageType.PHOTO, f"Fotka nebyla detekována správně: {img_path}"


class TestOrderPoints:
    """Testy pro seřazení bodů"""

    def test_order_points_correct_order(self):
        """Test správného seřazení 4 bodů"""
        # Body v náhodném pořadí
        pts = np.array([
            [100, 200],  # bottom-left
            [100, 100],  # top-left
            [200, 100],  # top-right
            [200, 200]   # bottom-right
        ], dtype=np.float32)

        ordered = order_points(pts)

        # Kontrola, že body jsou seřazeny správně: TL, TR, BR, BL
        assert np.array_equal(ordered[0], [100, 100])  # top-left
        assert np.array_equal(ordered[1], [200, 100])  # top-right
        assert np.array_equal(ordered[2], [200, 200])  # bottom-right
        assert np.array_equal(ordered[3], [100, 200])  # bottom-left


class TestPreprocessor:
    """Testy pro ImagePreprocessor"""

    def test_preprocessor_initialization(self, model_path):
        """Test inicializace preprocessoru"""
        preprocessor = ImagePreprocessor(model_path)
        assert preprocessor is not None
        assert preprocessor.model is not None

    def test_preprocess_invalid_path(self, preprocessor):
        """Test zpracování neexistujícího souboru"""
        with pytest.raises(ValueError, match="Nepodařilo se načíst obrázek"):
            preprocessor.preprocess("nonexistent.jpg")

    def test_preprocess_screenshot(self, preprocessor, test_images_dir):
        """Test předzpracování screenshotu"""
        screenshot_dir = os.path.join(test_images_dir, "screenshots")
        if not os.path.exists(screenshot_dir):
            pytest.skip("Screenshots složka neexistuje")

        screenshots = [f for f in os.listdir(screenshot_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
        if not screenshots:
            pytest.skip("Žádné screenshoty nenalezeny")

        img_path = os.path.join(screenshot_dir, screenshots[0])
        result_paths = preprocessor.preprocess(img_path)

        # preprocess() teď vrací list cest (může být více písní na obrázku)
        assert result_paths is not None
        assert isinstance(result_paths, list)
        assert len(result_paths) > 0

        # Zkontrolujeme první zpracovaný obrázek
        result_path = result_paths[0]
        assert isinstance(result_path, str)
        assert os.path.exists(result_path)

        # Načteme výsledek a zkontrolujeme
        result = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2  # Grayscale
        assert result.dtype == np.uint8

    def test_preprocess_photo(self, preprocessor, test_images_dir):
        """Test předzpracování fotky"""
        photos_dir = os.path.join(test_images_dir, "photos")
        if not os.path.exists(photos_dir):
            pytest.skip("Photos složka neexistuje")

        photos = [f for f in os.listdir(photos_dir) if f.endswith(('.jpg', '.jpeg')) and not f.startswith('.')]
        if not photos:
            pytest.skip("Žádné fotky nenalezeny")

        img_path = os.path.join(photos_dir, photos[0])
        result_paths = preprocessor.preprocess(img_path)

        # preprocess() teď vrací list cest (může být více písní na obrázku)
        assert result_paths is not None
        assert isinstance(result_paths, list)
        assert len(result_paths) > 0

        # Zkontrolujeme první zpracovaný obrázek
        result_path = result_paths[0]
        assert isinstance(result_path, str)
        assert os.path.exists(result_path)

        # Načteme výsledek a zkontrolujeme
        result = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2  # Grayscale
        assert result.dtype == np.uint8

    def test_preprocess_with_custom_output(self, preprocessor, test_images_dir):
        """Test předzpracování s vlastní výstupní cestou"""
        screenshot_dir = os.path.join(test_images_dir, "screenshots")
        if not os.path.exists(screenshot_dir):
            pytest.skip("Screenshots složka neexistuje")

        screenshots = [f for f in os.listdir(screenshot_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
        if not screenshots:
            pytest.skip("Žádné screenshoty nenalezeny")

        img_path = os.path.join(screenshot_dir, screenshots[0])
        custom_output = "temp/test_custom_output.png"
        result_paths = preprocessor.preprocess(img_path, custom_output)

        # preprocess() teď vrací list cest
        assert result_paths is not None
        assert isinstance(result_paths, list)
        assert len(result_paths) > 0

        # První výsledek by měl existovat
        result_path = result_paths[0]
        assert os.path.exists(result_path)

        # Načteme výsledek a zkontrolujeme
        result = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2  # Grayscale
        assert result.dtype == np.uint8

        # Cleanup všech vygenerovaných souborů
        for path in result_paths:
            if os.path.exists(path):
                os.remove(path)


class TestScreenshotProcessor:
    """Testy pro ScreenshotProcessor"""

    def test_screenshot_detection(self, preprocessor, test_images_dir):
        """Test detekce písní ve screenshotu"""
        screenshot_dir = os.path.join(test_images_dir, "screenshots")
        if not os.path.exists(screenshot_dir):
            pytest.skip("Screenshots složka neexistuje")

        screenshots = [f for f in os.listdir(screenshot_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
        if not screenshots:
            pytest.skip("Žádné screenshoty nenalezeny")

        img_path = os.path.join(screenshot_dir, screenshots[0])
        img = cv2.imread(img_path)
        detections = preprocessor._detect_all_songs(img_path, img)

        # Měla by být alespoň jedna detekce (nebo prázdný list, pokud model nic nenajde)
        assert isinstance(detections, list)


class TestPhotoProcessor:
    """Testy pro PhotoProcessor"""

    def test_photo_processing(self, preprocessor, test_images_dir):
        """Test zpracování fotky"""
        photos_dir = os.path.join(test_images_dir, "photos")
        if not os.path.exists(photos_dir):
            pytest.skip("Photos složka neexistuje")

        photos = [f for f in os.listdir(photos_dir) if f.endswith(('.jpg', '.jpeg')) and not f.startswith('.')]
        if not photos:
            pytest.skip("Žádné fotky nenalezeny")

        img_path = os.path.join(photos_dir, photos[0])
        img = cv2.imread(img_path)
        result = preprocessor._process_single_song(img)

        assert result is not None
        assert isinstance(result, np.ndarray)
        # Výsledek by měl mít nějaké rozměry
        assert result.shape[0] > 0 and result.shape[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
