"""
Unit tests for postprocessing functions
"""

import pytest
import numpy as np
import cv2

from paper_detection.segmentation.postprocess import (
    order_corners_clockwise,
    extract_corners_from_mask,
    mask_to_corners,
    scale_corners,
    min_corner_matching_error,
    clean_mask
)
from paper_detection.segmentation.dataset import generate_mask_from_corners


class TestOrderCornersClockwise:
    """Test corner ordering function"""

    def test_order_corners_basic(self):
        """Test basic corner ordering"""
        # Create corners in random order
        corners = np.array([
            [100, 100],  # TL
            [200, 100],  # TR
            [200, 200],  # BR
            [100, 200]   # BL
        ], dtype=np.float32)

        # Shuffle
        shuffled = corners[[2, 0, 3, 1]]

        # Order
        ordered = order_corners_clockwise(shuffled)

        # Should start with corner closest to (0, 0)
        assert ordered.shape == (4, 2)
        assert np.allclose(ordered[0], [100, 100], atol=1)

    def test_order_corners_invalid(self):
        """Test with invalid number of corners"""
        corners = np.array([[0, 0], [1, 1]], dtype=np.float32)
        with pytest.raises(ValueError):
            order_corners_clockwise(corners)


class TestMaskGeneration:
    """Test mask generation from corners"""

    def test_generate_mask_basic(self):
        """Test basic mask generation"""
        corners = np.array([
            [50, 50],
            [150, 50],
            [150, 150],
            [50, 150]
        ], dtype=np.float32)

        mask = generate_mask_from_corners(corners, (200, 200))

        # Check shape
        assert mask.shape == (200, 200)

        # Check that mask is binary
        assert set(np.unique(mask)).issubset({0, 255})

        # Check that some pixels are filled
        assert np.sum(mask == 255) > 0

        # Check that corners are inside
        for corner in corners.astype(int):
            x, y = corner
            if 0 <= y < 200 and 0 <= x < 200:
                assert mask[y, x] == 255

    def test_generate_mask_empty(self):
        """Test mask generation with empty region"""
        corners = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ], dtype=np.float32)

        mask = generate_mask_from_corners(corners, (100, 100))
        assert mask.shape == (100, 100)


class TestCornerExtraction:
    """Test corner extraction from masks"""

    def test_extract_corners_square(self):
        """Test corner extraction from square mask"""
        # Create a square mask
        mask = np.zeros((200, 200), dtype=np.uint8)
        corners_gt = np.array([
            [50, 50],
            [150, 50],
            [150, 150],
            [50, 150]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [corners_gt], 255)

        # Extract corners
        corners = extract_corners_from_mask(mask, min_contour_area=100)

        # Should get 4 corners
        assert corners is not None
        assert corners.shape == (4, 2)

    def test_extract_corners_empty_mask(self):
        """Test corner extraction from empty mask"""
        mask = np.zeros((200, 200), dtype=np.uint8)

        corners = extract_corners_from_mask(mask)

        # Should return None for empty mask
        assert corners is None

    def test_extract_corners_small_contour(self):
        """Test corner extraction with small contour (below threshold)"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Create tiny square
        cv2.rectangle(mask, (100, 100), (105, 105), 255, -1)

        corners = extract_corners_from_mask(mask, min_contour_area=1000)

        # Should return None because area is too small
        assert corners is None


class TestMaskToCorners:
    """Test full pipeline from probability mask to corners"""

    def test_mask_to_corners_basic(self):
        """Test basic mask to corners conversion"""
        # Create probability mask
        mask_prob = np.zeros((200, 200), dtype=np.float32)
        corners_gt = np.array([
            [50, 50],
            [150, 50],
            [150, 150],
            [50, 150]
        ], dtype=np.int32)
        cv2.fillPoly(mask_prob, [corners_gt], 1.0)

        # Convert to corners
        corners = mask_to_corners(mask_prob, threshold=0.5)

        # Should get 4 corners
        assert corners is not None
        assert corners.shape == (4, 2)


class TestScaleCorners:
    """Test corner scaling"""

    def test_scale_corners_basic(self):
        """Test basic corner scaling"""
        corners = np.array([
            [50, 50],
            [150, 50],
            [150, 150],
            [50, 150]
        ], dtype=np.float32)

        # Scale from 200x200 to 400x400
        scaled = scale_corners(corners, (200, 200), (400, 400))

        expected = corners * 2
        assert np.allclose(scaled, expected)

    def test_scale_corners_non_uniform(self):
        """Test non-uniform scaling"""
        corners = np.array([
            [100, 50],
            [200, 50],
            [200, 150],
            [100, 150]
        ], dtype=np.float32)

        # Scale from 200x200 to 400x100
        scaled = scale_corners(corners, (200, 200), (400, 100))

        # x should be doubled, y should be halved
        assert np.allclose(scaled[:, 0], corners[:, 0] * 2)
        assert np.allclose(scaled[:, 1], corners[:, 1] * 0.5)


class TestMinCornerMatchingError:
    """Test corner matching error calculation"""

    def test_matching_identical(self):
        """Test matching with identical corners"""
        corners = np.array([
            [50, 50],
            [150, 50],
            [150, 150],
            [50, 150]
        ], dtype=np.float32)

        error, matched = min_corner_matching_error(corners, corners)

        # Error should be zero
        assert error == 0.0
        assert np.allclose(matched, corners)

    def test_matching_rotated(self):
        """Test matching with rotated corners"""
        corners1 = np.array([
            [50, 50],
            [150, 50],
            [150, 150],
            [50, 150]
        ], dtype=np.float32)

        # Rotate by 1 position
        corners2 = np.roll(corners1, 1, axis=0)

        error, matched = min_corner_matching_error(corners1, corners2)

        # Error should be zero
        assert error == 0.0

    def test_matching_reversed(self):
        """Test matching with reversed corners (CW vs CCW)"""
        corners1 = np.array([
            [50, 50],
            [150, 50],
            [150, 150],
            [50, 150]
        ], dtype=np.float32)

        # Reverse order
        corners2 = corners1[::-1]

        error, matched = min_corner_matching_error(corners1, corners2)

        # Error should be zero
        assert error == 0.0


class TestCleanMask:
    """Test mask cleaning"""

    def test_clean_mask_basic(self):
        """Test basic mask cleaning"""
        # Create noisy mask
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)

        # Add noise
        mask[10, 10] = 255
        mask[180, 180] = 255

        # Clean
        cleaned = clean_mask(mask)

        # Should remove small noise
        assert cleaned.shape == mask.shape
        assert np.sum(cleaned) < np.sum(mask)
