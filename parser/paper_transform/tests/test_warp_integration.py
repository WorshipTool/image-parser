"""
Integration tests for warp_paper with real images
Tests transformation against reference images
"""

import json
import pytest
import cv2
import numpy as np
from pathlib import Path

from paper_transform import warp_paper


# Paths
TEST_DATA_DIR = Path(__file__).parent / "test_data"
REFERENCES_DIR = TEST_DATA_DIR / "warp_references"
IMAGES_DIR = TEST_DATA_DIR
CORNERS_FILE = TEST_DATA_DIR / "test_corners.json"

# Test cases
TEST_CASES = [
    "IMG_20230826_092429.jpg",
    "IMG_20230826_093159.jpg",
    "test_image_2.jpg",
]


def denormalize_corners(corners_norm, img_shape):
    """Convert normalized corners (0-1) to pixel coordinates"""
    h, w = img_shape[:2]
    corners_px = np.array(corners_norm, dtype=np.float32)
    corners_px[:, 0] *= w
    corners_px[:, 1] *= h
    return corners_px


def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two images"""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)


def calculate_similarity_percentage(img1, img2):
    """Calculate similarity percentage (100% = identical)"""
    max_pixel_value = 255.0
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return 100.0
    # Convert MSE to similarity percentage
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    # PSNR > 40 is excellent, normalize to percentage
    similarity = min(100.0, (psnr / 50.0) * 100.0)
    return similarity


@pytest.fixture
def corners_data():
    """Load corners from JSON"""
    with open(CORNERS_FILE) as f:
        return json.load(f)


@pytest.mark.parametrize("image_name", TEST_CASES)
def test_warp_against_reference(image_name, corners_data):
    """Test warp_paper produces same result as reference"""

    # Load source image
    img_path = IMAGES_DIR / image_name
    image = cv2.imread(str(img_path))
    assert image is not None, f"Failed to load {img_path}"

    # Get corners
    assert image_name in corners_data, f"No corners for {image_name}"
    corners_norm = corners_data[image_name]["corners"]
    corners_px = denormalize_corners(corners_norm, image.shape)

    # Apply warp
    warped = warp_paper(image, corners_px)

    # Load reference
    ref_name = f"{Path(image_name).stem}_warped.jpg"
    ref_path = REFERENCES_DIR / ref_name
    reference = cv2.imread(str(ref_path))
    assert reference is not None, f"Reference not found: {ref_path}"

    # Compare dimensions
    assert warped.shape == reference.shape, \
        f"Shape mismatch: warped={warped.shape}, reference={reference.shape}"

    # Calculate similarity
    mse = calculate_mse(warped, reference)
    similarity = calculate_similarity_percentage(warped, reference)

    print(f"\n{image_name}:")
    print(f"  MSE: {mse:.2f}")
    print(f"  Similarity: {similarity:.1f}%")

    # Should be identical or very close (allowing for JPEG compression)
    assert mse < 10.0, \
        f"Images too different (MSE={mse:.2f}). Expected MSE < 10.0"


def test_all_references_exist():
    """Verify all reference images exist"""
    for image_name in TEST_CASES:
        ref_name = f"{Path(image_name).stem}_warped.jpg"
        ref_path = REFERENCES_DIR / ref_name
        assert ref_path.exists(), f"Missing reference: {ref_path}"


def test_warp_consistency():
    """Test that warping the same image twice gives identical results"""

    # Use first test image
    image_name = TEST_CASES[0]

    with open(CORNERS_FILE) as f:
        corners_data = json.load(f)

    # Load image
    img_path = IMAGES_DIR / image_name
    image = cv2.imread(str(img_path))

    # Get corners
    corners_norm = corners_data[image_name]["corners"]
    corners_px = denormalize_corners(corners_norm, image.shape)

    # Warp twice
    warped1 = warp_paper(image, corners_px)
    warped2 = warp_paper(image, corners_px)

    # Should be identical
    mse = calculate_mse(warped1, warped2)
    assert mse == 0.0, f"Warp not consistent (MSE={mse})"
