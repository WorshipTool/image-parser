"""
Simple tests for orient_by_text function using ground truth data
"""

import json
import pytest
import cv2
from pathlib import Path

from paper_transform.document_orient.orient_by_text import orient_by_text


# Path to test data
TEST_DATA_DIR = Path(__file__).parent / "test_data"
GROUND_TRUTH_FILE = TEST_DATA_DIR / "ground_truth.json"


def normalize_angle(angle: int) -> int:
    """Normalize angle to 0, 90, 180, 270 range"""
    angle = angle % 360
    if angle < 0:
        angle += 360
    return angle


@pytest.fixture
def ground_truth_data():
    """Load ground truth data from JSON file"""
    with open(GROUND_TRUTH_FILE, 'r') as f:
        return json.load(f)


def test_orient_by_text_with_ground_truth(ground_truth_data):
    """Test orient_by_text against all images in ground_truth.json"""

    for image_name, expected_data in ground_truth_data.items():
        image_path = TEST_DATA_DIR / image_name
        expected_rotation = normalize_angle(expected_data["expected_rotation"])

        # Load image
        image = cv2.imread(str(image_path))
        assert image is not None, f"Failed to load image: {image_path}"

        # Run orient_by_text
        result = orient_by_text(image)

        # Check that we got a result
        assert result is not None, f"orient_by_text returned None for {image_name}"

        _, detected_angle, confidence = result

        # Check that angle matches expected
        assert detected_angle == expected_rotation, \
            f"{image_name}: Expected {expected_rotation}°, got {detected_angle}°"

        # Check that confidence is reasonable
        assert confidence > 0, f"{image_name}: Confidence should be > 0, got {confidence}"

        print(f"✓ {image_name}: angle={detected_angle}°, confidence={confidence:.2f}")


@pytest.mark.parametrize("image_name,expected_rotation", [
    ("test_orient_00.jpg", 180),
    ("test_orient_01.jpg", 90),
    ("test_orient_02.jpg", 0),
    ("test_orient_03.jpg", 270),  # -90 normalized to 270
    ("test_orient_04.jpg", 0),
])
def test_orient_by_text_individual(image_name, expected_rotation):
    """Test individual images with orient_by_text"""

    image_path = TEST_DATA_DIR / image_name

    # Load image
    image = cv2.imread(str(image_path))
    assert image is not None, f"Failed to load image: {image_path}"

    # Run orient_by_text
    result = orient_by_text(image)

    # Check result
    assert result is not None, f"orient_by_text returned None for {image_name}"

    _, detected_angle, confidence = result

    # Verify angle
    assert detected_angle == expected_rotation, \
        f"Expected {expected_rotation}°, got {detected_angle}°"

    # Verify confidence
    assert confidence > 0, f"Confidence should be positive"
