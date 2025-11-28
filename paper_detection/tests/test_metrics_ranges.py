"""
Tests for paper detection metrics with expected ranges
"""

import cv2
import numpy as np
import pytest
from pathlib import Path
from paper_detection import PaperDetector, PaperVisualizer


# Expected metrics for each test image with tolerances
# Format: (image_name, detection_config, expected_metrics, tolerances)
TEST_IMAGE_METRICS = [
    (
        'IMG_20230826_092914.jpg',
        {'brightness_threshold': 180, 'min_area_ratio': 0.005},
        {
            'cover_ratio': 0.0990,
            'rectangularity': 0.4940,
            'angle': 59.50,
            'perspective_angle': 27.70,
        },
        {
            'cover_ratio': 0.02,  # ±2% tolerance
            'rectangularity': 0.15,  # ±15% tolerance (low rect due to perspective)
            'angle': 10.0,  # ±10 degrees
            'perspective_angle': 5.0,  # ±5 degrees
        }
    ),
    (
        'IMG_20230826_093159.jpg',
        {'brightness_threshold': 200, 'min_area_ratio': 0.01},
        {
            'cover_ratio': 0.2503,
            'rectangularity': 0.9730,
            'angle': 19.94,
            'perspective_angle': 1.56,
        },
        {
            'cover_ratio': 0.02,
            'rectangularity': 0.05,
            'angle': 5.0,
            'perspective_angle': 1.0,
        }
    ),
    (
        'test_image_1.jpeg',
        {'brightness_threshold': 180, 'min_area_ratio': 0.005},
        {
            'cover_ratio': 0.1797,
            'rectangularity': 0.8759,
            'angle': 107.50,
            'perspective_angle': 6.65,
        },
        {
            'cover_ratio': 0.02,
            'rectangularity': 0.1,
            'angle': 5.0,
            'perspective_angle': 2.0,
        }
    ),
    (
        'test_image_2.jpg',
        {'brightness_threshold': 50, 'min_area_ratio': 0.003},
        {
            'cover_ratio': 0.9398,
            'rectangularity': 0.9403,
            'angle': 90.00,
            'perspective_angle': 0.50,
        },
        {
            'cover_ratio': 0.05,
            'rectangularity': 0.05,
            'angle': 5.0,
            'perspective_angle': 1.0,
        }
    ),
    (
        'test_image_3.jpeg',
        {'brightness_threshold': 120, 'min_area_ratio': 0.003},
        {
            'cover_ratio': 0.4718,
            'rectangularity': 0.8799,
            'angle': 177.50,
            'perspective_angle': 6.65,
        },
        {
            'cover_ratio': 0.03,
            'rectangularity': 0.1,
            'angle': 5.0,
            'perspective_angle': 2.0,
        }
    ),
]


class TestMetricsRanges:
    """Test that metrics are within expected ranges for specific test images"""

    @pytest.fixture
    def test_images_dir(self):
        """Get test images directory"""
        return Path(__file__).parent / "test_images"

    @pytest.mark.parametrize("image_name,config,expected,tolerances", TEST_IMAGE_METRICS)
    def test_metrics_in_expected_range(self, test_images_dir, image_name, config, expected, tolerances):
        """Test that all metrics are within expected ranges"""
        image_path = test_images_dir / image_name

        # Load image
        assert image_path.exists(), f"Image not found: {image_path}"
        image = cv2.imread(str(image_path))
        assert image is not None, f"Failed to load image: {image_path}"

        # Detect paper
        detector = PaperDetector(**config)
        corners = detector.detect(image)
        assert corners is not None, f"Paper not detected in {image_name}"

        # Get metrics
        metrics = detector.get_paper_metrics(corners, image.shape)

        # Verify each metric is within tolerance
        for metric_name, expected_value in expected.items():
            actual_value = metrics[metric_name]
            tolerance = tolerances[metric_name]

            min_value = expected_value - tolerance
            max_value = expected_value + tolerance

            assert min_value <= actual_value <= max_value, \
                f"{image_name}: {metric_name} = {actual_value:.4f}, " \
                f"expected {expected_value:.4f} ± {tolerance:.4f} " \
                f"(range: {min_value:.4f} to {max_value:.4f})"

        print(f"✓ {image_name}: All metrics within expected ranges")

    @pytest.mark.parametrize("image_name,config,expected,tolerances", TEST_IMAGE_METRICS)
    def test_cover_ratio_range(self, test_images_dir, image_name, config, expected, tolerances):
        """Test cover_ratio is in expected range"""
        image_path = test_images_dir / image_name

        if not image_path.exists():
            pytest.skip(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            pytest.skip(f"Failed to load image: {image_path}")

        detector = PaperDetector(**config)
        corners = detector.detect(image)

        if corners is None:
            pytest.skip(f"Paper not detected in {image_name}")

        metrics = detector.get_paper_metrics(corners, image.shape)

        expected_value = expected['cover_ratio']
        tolerance = tolerances['cover_ratio']
        actual_value = metrics['cover_ratio']

        assert expected_value - tolerance <= actual_value <= expected_value + tolerance, \
            f"{image_name}: cover_ratio={actual_value:.4f}, expected {expected_value:.4f}±{tolerance:.4f}"

    @pytest.mark.parametrize("image_name,config,expected,tolerances", TEST_IMAGE_METRICS)
    def test_rectangularity_range(self, test_images_dir, image_name, config, expected, tolerances):
        """Test rectangularity is in expected range"""
        image_path = test_images_dir / image_name

        if not image_path.exists():
            pytest.skip(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            pytest.skip(f"Failed to load image: {image_path}")

        detector = PaperDetector(**config)
        corners = detector.detect(image)

        if corners is None:
            pytest.skip(f"Paper not detected in {image_name}")

        metrics = detector.get_paper_metrics(corners, image.shape)

        expected_value = expected['rectangularity']
        tolerance = tolerances['rectangularity']
        actual_value = metrics['rectangularity']

        assert expected_value - tolerance <= actual_value <= expected_value + tolerance, \
            f"{image_name}: rectangularity={actual_value:.4f}, expected {expected_value:.4f}±{tolerance:.4f}"

    @pytest.mark.parametrize("image_name,config,expected,tolerances", TEST_IMAGE_METRICS)
    def test_angle_range(self, test_images_dir, image_name, config, expected, tolerances):
        """Test angle is in expected range"""
        image_path = test_images_dir / image_name

        if not image_path.exists():
            pytest.skip(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            pytest.skip(f"Failed to load image: {image_path}")

        detector = PaperDetector(**config)
        corners = detector.detect(image)

        if corners is None:
            pytest.skip(f"Paper not detected in {image_name}")

        metrics = detector.get_paper_metrics(corners, image.shape)

        expected_value = expected['angle']
        tolerance = tolerances['angle']
        actual_value = metrics['angle']

        assert expected_value - tolerance <= actual_value <= expected_value + tolerance, \
            f"{image_name}: angle={actual_value:.2f}°, expected {expected_value:.2f}°±{tolerance:.2f}°"

    @pytest.mark.parametrize("image_name,config,expected,tolerances", TEST_IMAGE_METRICS)
    def test_perspective_angle_range(self, test_images_dir, image_name, config, expected, tolerances):
        """Test perspective_angle is in expected range"""
        image_path = test_images_dir / image_name

        if not image_path.exists():
            pytest.skip(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            pytest.skip(f"Failed to load image: {image_path}")

        detector = PaperDetector(**config)
        corners = detector.detect(image)

        if corners is None:
            pytest.skip(f"Paper not detected in {image_name}")

        metrics = detector.get_paper_metrics(corners, image.shape)

        expected_value = expected['perspective_angle']
        tolerance = tolerances['perspective_angle']
        actual_value = metrics['perspective_angle']

        assert expected_value - tolerance <= actual_value <= expected_value + tolerance, \
            f"{image_name}: perspective_angle={actual_value:.2f}°, expected {expected_value:.2f}°±{tolerance:.2f}°"

    @pytest.mark.parametrize("image_name,config,expected,tolerances", TEST_IMAGE_METRICS)
    def test_visualization_with_metrics(self, test_images_dir, image_name, config, expected, tolerances):
        """Test that visualization with metrics works for each image"""
        image_path = test_images_dir / image_name

        if not image_path.exists():
            pytest.skip(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            pytest.skip(f"Failed to load image: {image_path}")

        detector = PaperDetector(**config)
        corners = detector.detect(image)

        if corners is None:
            pytest.skip(f"Paper not detected in {image_name}")

        metrics = detector.get_paper_metrics(corners, image.shape)

        visualizer = PaperVisualizer()
        result = visualizer.visualize_with_metrics(image, corners, metrics)

        assert result is not None
        assert result.shape == image.shape
        assert not np.array_equal(result, image)

        # Save for manual inspection
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"metrics_range_{image_name}"
        cv2.imwrite(str(output_path), result)
        assert output_path.exists()
