"""
Unit tests for warp module
"""

import pytest
import numpy as np
import cv2
from paper_transform.warp import warp_paper, _order_corners, _compute_output_dimensions


class TestOrderCorners:
    """Test corner ordering functionality"""

    def test_already_ordered_corners(self):
        """Test that already ordered corners remain in correct order"""
        corners = np.array([
            [0, 0],      # top-left
            [100, 0],    # top-right
            [100, 100],  # bottom-right
            [0, 100]     # bottom-left
        ], dtype=np.float32)

        ordered = _order_corners(corners)

        np.testing.assert_array_almost_equal(ordered, corners)

    def test_unordered_corners(self):
        """Test that unordered corners are properly sorted"""
        # Corners in random order
        corners = np.array([
            [100, 100],  # bottom-right
            [0, 0],      # top-left
            [0, 100],    # bottom-left
            [100, 0]     # top-right
        ], dtype=np.float32)

        ordered = _order_corners(corners)

        expected = np.array([
            [0, 0],      # top-left
            [100, 0],    # top-right
            [100, 100],  # bottom-right
            [0, 100]     # bottom-left
        ], dtype=np.float32)

        np.testing.assert_array_almost_equal(ordered, expected)

    def test_rotated_rectangle(self):
        """Test corner ordering for a rotated rectangle"""
        corners = np.array([
            [50, 10],   # top-right (rotated)
            [90, 50],   # bottom-right
            [50, 90],   # bottom-left
            [10, 50]    # top-left
        ], dtype=np.float32)

        ordered = _order_corners(corners)

        # Should be ordered: TL, TR, BR, BL
        assert ordered[0][1] < ordered[2][1]  # top-left y < bottom-right y
        assert ordered[0][0] < ordered[1][0]  # top-left x < top-right x


class TestComputeOutputDimensions:
    """Test output dimension computation"""

    def test_square(self):
        """Test dimensions for a square"""
        corners = np.array([
            [0, 0],
            [100, 0],
            [100, 100],
            [0, 100]
        ], dtype=np.float32)

        width, height = _compute_output_dimensions(corners)

        assert width == 100
        assert height == 100

    def test_rectangle(self):
        """Test dimensions for a rectangle"""
        corners = np.array([
            [0, 0],
            [200, 0],
            [200, 100],
            [0, 100]
        ], dtype=np.float32)

        width, height = _compute_output_dimensions(corners)

        assert width == 200
        assert height == 100

    def test_trapezoid(self):
        """Test dimensions for a perspective-distorted rectangle"""
        corners = np.array([
            [10, 0],     # top-left
            [190, 0],    # top-right
            [200, 100],  # bottom-right
            [0, 100]     # bottom-left
        ], dtype=np.float32)

        width, height = _compute_output_dimensions(corners)

        # Should use maximum edge length
        assert width >= 180  # at least the top edge
        assert height == 100


class TestWarpPaper:
    """Test warp_paper function"""

    def setup_method(self):
        """Create a test image before each test"""
        # Create a simple 200x200 white image with a black rectangle
        self.test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.rectangle(self.test_image, (50, 50), (150, 150), (0, 0, 0), 2)

    def test_warp_no_distortion(self):
        """Test warping with no perspective distortion"""
        corners = np.array([
            [50, 50],
            [150, 50],
            [150, 150],
            [50, 150]
        ], dtype=np.float32)

        warped = warp_paper(self.test_image, corners)

        # Should produce a 100x100 image
        assert warped.shape[0] == 100
        assert warped.shape[1] == 100
        assert warped.shape[2] == 3

    def test_warp_with_custom_size(self):
        """Test warping with custom output size"""
        corners = np.array([
            [50, 50],
            [150, 50],
            [150, 150],
            [50, 150]
        ], dtype=np.float32)

        warped = warp_paper(self.test_image, corners, dst_size=(200, 300))

        assert warped.shape[0] == 300  # height
        assert warped.shape[1] == 200  # width

    def test_warp_invalid_image(self):
        """Test error handling for invalid image"""
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        with pytest.raises(ValueError, match="Input image is None or empty"):
            warp_paper(None, corners)

    def test_warp_invalid_corners(self):
        """Test error handling for invalid corners"""
        with pytest.raises(ValueError, match="Corners must be an array of 4 points"):
            warp_paper(self.test_image, np.array([[0, 0], [100, 0]]))

        with pytest.raises(ValueError, match="Corners must be an array of 4 points"):
            warp_paper(self.test_image, None)

    def test_warp_preserves_color(self):
        """Test that warping preserves color channels"""
        corners = np.array([
            [0, 0],
            [199, 0],
            [199, 199],
            [0, 199]
        ], dtype=np.float32)

        warped = warp_paper(self.test_image, corners)

        # Check that we have 3 color channels
        assert len(warped.shape) == 3
        assert warped.shape[2] == 3
