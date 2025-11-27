"""
Tests for paper_detection module
"""

import cv2
import numpy as np
import pytest
import os
from pathlib import Path

from paper_detection.detector import PaperDetector
from paper_detection.visualizer import PaperVisualizer


class TestPaperDetector:
    """Tests for PaperDetector"""

    @pytest.fixture
    def detector(self):
        """Create detector instance for tests"""
        return PaperDetector()

    @pytest.fixture
    def test_image_path(self):
        """Path to test image"""
        # Relative path from project root
        base_path = Path(__file__).parent.parent.parent
        return base_path / "images" / "photos" / "IMG_20230826_093159.jpg"

    @pytest.fixture
    def test_image(self, test_image_path):
        """Load test image"""
        if not test_image_path.exists():
            pytest.skip(f"Test image not found: {test_image_path}")
        return cv2.imread(str(test_image_path))

    @pytest.fixture(params=['test_image_1.jpeg', 'test_image_2.jpg', 'test_image_3.jpeg'])
    def test_images_new(self, request):
        """Load new test images from test_images folder"""
        test_images_dir = Path(__file__).parent / "test_images"
        image_path = test_images_dir / request.param

        if not image_path.exists():
            pytest.skip(f"Test image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            pytest.skip(f"Failed to load image: {image_path}")

        return image, request.param

    def test_detector_init(self):
        """Test detector initialization"""
        detector = PaperDetector()
        assert detector.brightness_threshold == 200
        assert detector.min_area_ratio == 0.01
        assert detector.approx_epsilon == 0.02
        assert detector.use_threshold_method == True

    def test_detector_custom_params(self):
        """Test initialization with custom parameters"""
        detector = PaperDetector(
            brightness_threshold=180,
            min_area_ratio=0.1,
            approx_epsilon=0.03,
            use_threshold_method=False
        )
        assert detector.brightness_threshold == 180
        assert detector.min_area_ratio == 0.1
        assert detector.approx_epsilon == 0.03
        assert detector.use_threshold_method == False

    def test_detect_none_image(self, detector):
        """Test detection with None image"""
        result = detector.detect(None)
        assert result is None

    def test_detect_empty_image(self, detector):
        """Test detection with empty image"""
        empty_image = np.array([])
        result = detector.detect(empty_image)
        assert result is None

    def test_detect_paper(self, detector, test_image):
        """Test paper detection on real image"""
        corners = detector.detect(test_image)

        # Verify that corners were found
        assert corners is not None, "Paper was not detected"
        assert corners.shape == (4, 2), "4 corners were not found"

        # Verify that corners are valid coordinates
        assert np.all(corners >= 0), "Corners contain negative coordinates"
        assert np.all(corners[:, 0] < test_image.shape[1]), "X coordinates out of range"
        assert np.all(corners[:, 1] < test_image.shape[0]), "Y coordinates out of range"

    def test_detect_paper_ordering(self, detector, test_image):
        """Test correct corner ordering"""
        corners = detector.detect(test_image)

        if corners is None:
            pytest.skip("Paper was not detected")

        # Top-left should have smallest sum of x+y
        # Bottom-right should have largest sum of x+y
        tl_sum = corners[0][0] + corners[0][1]
        br_sum = corners[2][0] + corners[2][1]
        assert br_sum > tl_sum, "Corners are not correctly ordered"

    def test_get_paper_dimensions(self, detector, test_image):
        """Test paper dimensions calculation"""
        corners = detector.detect(test_image)

        if corners is None:
            pytest.skip("Paper was not detected")

        width, height = detector.get_paper_dimensions(corners)

        assert width > 0, "Width must be positive"
        assert height > 0, "Height must be positive"
        assert isinstance(width, int), "Width must be integer"
        assert isinstance(height, int), "Height must be integer"

    def test_detect_with_different_thresholds(self, test_image):
        """Test detection with different thresholds"""
        # Lower brightness threshold (detects darker areas too)
        low_threshold_detector = PaperDetector(
            brightness_threshold=150
        )
        corners1 = low_threshold_detector.detect(test_image)

        # Higher brightness threshold (only very bright areas)
        high_threshold_detector = PaperDetector(
            brightness_threshold=220
        )
        corners2 = high_threshold_detector.detect(test_image)

        # At least one should find the paper
        assert corners1 is not None or corners2 is not None

    def test_detect_on_new_test_images(self, test_images_new):
        """Test detection on new test images with adaptive parameters"""
        image, image_name = test_images_new

        # Try different detection strategies based on image characteristics
        # Some papers may require different brightness thresholds
        detection_strategies = [
            # Strategy 1: Standard threshold (works for bright, clean papers)
            {'brightness_threshold': 200, 'min_area_ratio': 0.01},
            # Strategy 2: Lower threshold (works for slightly darker papers)
            {'brightness_threshold': 180, 'min_area_ratio': 0.005},
            # Strategy 3: Very low threshold (works for shadowed papers)
            {'brightness_threshold': 120, 'min_area_ratio': 0.003, 'approx_epsilon': 0.03},
            # Strategy 4: Minimal threshold (last resort)
            {'brightness_threshold': 100, 'min_area_ratio': 0.003, 'approx_epsilon': 0.03},
            # Strategy 5: Canny edge detection (works well for clear edges)
            {'use_threshold_method': False, 'min_area_ratio': 0.005},
        ]

        corners = None
        successful_strategy = None

        for i, strategy_params in enumerate(detection_strategies):
            detector = PaperDetector(**strategy_params)
            corners = detector.detect(image)

            if corners is not None:
                successful_strategy = i + 1
                break

        # Verify that paper was detected with at least one strategy
        assert corners is not None, f"Paper was not detected in {image_name} with any strategy"
        assert corners.shape == (4, 2), f"4 corners were not found in {image_name}"

        # Verify valid coordinates
        assert np.all(corners >= 0), f"Corners contain negative coordinates in {image_name}"
        assert np.all(corners[:, 0] < image.shape[1]), f"X coordinates out of range in {image_name}"
        assert np.all(corners[:, 1] < image.shape[0]), f"Y coordinates out of range in {image_name}"

        print(f"  ✓ {image_name} detected with strategy {successful_strategy}")

    def test_visualize_new_test_images(self, test_images_new):
        """Test visualization on new test images"""
        image, image_name = test_images_new

        # Use adaptive detection
        detection_strategies = [
            {'brightness_threshold': 180, 'min_area_ratio': 0.005},
            {'brightness_threshold': 120, 'min_area_ratio': 0.003, 'approx_epsilon': 0.03},
            {'brightness_threshold': 100, 'min_area_ratio': 0.003, 'approx_epsilon': 0.03},
        ]

        corners = None
        for strategy_params in detection_strategies:
            detector = PaperDetector(**strategy_params)
            corners = detector.detect(image)
            if corners is not None:
                break

        if corners is None:
            pytest.skip(f"Paper was not detected in {image_name}")

        # Visualize
        visualizer = PaperVisualizer()
        result = visualizer.visualize(image, corners)

        assert result is not None, f"Visualization failed for {image_name}"
        assert result.shape == image.shape, f"Result shape mismatch for {image_name}"
        assert not np.array_equal(result, image), f"Result unchanged for {image_name}"

        # Save visualization for manual inspection
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"new_{image_name}"
        cv2.imwrite(str(output_path), result)
        assert output_path.exists(), f"Failed to save visualization for {image_name}"


class TestPaperVisualizer:
    """Tests for PaperVisualizer"""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance for tests"""
        return PaperVisualizer()

    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return PaperDetector()

    @pytest.fixture
    def test_image_path(self):
        """Path to test image"""
        base_path = Path(__file__).parent.parent.parent
        return base_path / "images" / "photos" / "IMG_20230826_093159.jpg"

    @pytest.fixture
    def test_image(self, test_image_path):
        """Load test image"""
        if not test_image_path.exists():
            pytest.skip(f"Test image not found: {test_image_path}")
        return cv2.imread(str(test_image_path))

    @pytest.fixture
    def test_corners(self, detector, test_image):
        """Detect paper corners on test image"""
        corners = detector.detect(test_image)
        if corners is None:
            pytest.skip("Paper was not detected")
        return corners

    def test_visualizer_init(self):
        """Test visualizer initialization"""
        visualizer = PaperVisualizer()
        assert visualizer.border_color == (255, 100, 0)
        assert visualizer.border_thickness == 3
        assert visualizer.overlay_color == (255, 200, 100)
        assert visualizer.overlay_alpha == 0.3

    def test_visualizer_custom_params(self):
        """Test initialization with custom parameters"""
        visualizer = PaperVisualizer(
            border_color=(0, 255, 0),
            border_thickness=5,
            overlay_color=(0, 200, 255),
            overlay_alpha=0.5
        )
        assert visualizer.border_color == (0, 255, 0)
        assert visualizer.border_thickness == 5
        assert visualizer.overlay_color == (0, 200, 255)
        assert visualizer.overlay_alpha == 0.5

    def test_visualize_none_image(self, visualizer):
        """Test visualization with None image"""
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        result = visualizer.visualize(None, corners)
        assert result is None

    def test_visualize_none_corners(self, visualizer, test_image):
        """Test visualization with None corners"""
        result = visualizer.visualize(test_image, None)
        # Should return original image
        assert result is not None
        assert result.shape == test_image.shape

    def test_visualize_basic(self, visualizer, test_image, test_corners):
        """Test basic visualization"""
        result = visualizer.visualize(test_image, test_corners)

        assert result is not None
        assert result.shape == test_image.shape
        assert result.dtype == test_image.dtype

        # Visualized image should differ from original
        assert not np.array_equal(result, test_image)

    def test_visualize_border_only(self, visualizer, test_image, test_corners):
        """Test visualization with frame only"""
        result = visualizer.visualize(
            test_image,
            test_corners,
            draw_border=True,
            draw_overlay=False
        )

        assert result is not None
        assert not np.array_equal(result, test_image)

    def test_visualize_overlay_only(self, visualizer, test_image, test_corners):
        """Test visualization with overlay only"""
        result = visualizer.visualize(
            test_image,
            test_corners,
            draw_border=False,
            draw_overlay=True
        )

        assert result is not None
        assert not np.array_equal(result, test_image)

    def test_visualize_with_info(self, visualizer, test_image, test_corners):
        """Test visualization with information"""
        result = visualizer.visualize_with_info(
            test_image,
            test_corners,
            show_dimensions=True
        )

        assert result is not None
        assert result.shape == test_image.shape
        assert not np.array_equal(result, test_image)

    def test_create_side_by_side(self, visualizer, test_image, test_corners):
        """Test creating comparison image"""
        visualized = visualizer.visualize(test_image, test_corners)
        result = visualizer.create_side_by_side(test_image, visualized)

        assert result is not None
        # Width should be approximately 2x larger
        assert result.shape[1] >= test_image.shape[1] * 1.8
        assert result.shape[0] == test_image.shape[0]

    def test_create_side_by_side_none_images(self, visualizer):
        """Test creating comparison image with None inputs"""
        result = visualizer.create_side_by_side(None, None)
        assert result is None


class TestIntegration:
    """Integration tests"""

    @pytest.fixture
    def test_image_path(self):
        """Path to test image"""
        base_path = Path(__file__).parent.parent.parent
        return base_path / "images" / "photos" / "IMG_20230826_093159.jpg"

    def test_full_pipeline(self, test_image_path):
        """Test complete pipeline: detection + visualization"""
        # Load image
        if not test_image_path.exists():
            pytest.skip(f"Test image not found: {test_image_path}")

        image = cv2.imread(str(test_image_path))
        assert image is not None, "Failed to load image"

        # Detect paper
        detector = PaperDetector()
        corners = detector.detect(image)
        assert corners is not None, "Paper was not detected"

        # Visualize
        visualizer = PaperVisualizer()
        result = visualizer.visualize(image, corners)
        assert result is not None
        assert not np.array_equal(result, image)

        # Save result (for manual review)
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_result.jpg"
        cv2.imwrite(str(output_path), result)
        assert output_path.exists(), "Output image was not saved"

    def test_full_pipeline_with_info(self, test_image_path):
        """Test complete pipeline with information"""
        if not test_image_path.exists():
            pytest.skip(f"Test image not found: {test_image_path}")

        image = cv2.imread(str(test_image_path))
        detector = PaperDetector()
        corners = detector.detect(image)

        if corners is None:
            pytest.skip("Paper was not detected")

        visualizer = PaperVisualizer()
        result = visualizer.visualize_with_info(image, corners)

        # Save result
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_result_with_info.jpg"
        cv2.imwrite(str(output_path), result)
        assert output_path.exists()

    def test_side_by_side_comparison(self, test_image_path):
        """Test original vs processed image comparison"""
        if not test_image_path.exists():
            pytest.skip(f"Test image not found: {test_image_path}")

        image = cv2.imread(str(test_image_path))
        detector = PaperDetector()
        corners = detector.detect(image)

        if corners is None:
            pytest.skip("Paper was not detected")

        visualizer = PaperVisualizer()
        visualized = visualizer.visualize(image, corners)
        comparison = visualizer.create_side_by_side(image, visualized)

        # Save result
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_comparison.jpg"
        cv2.imwrite(str(output_path), comparison)
        assert output_path.exists()
