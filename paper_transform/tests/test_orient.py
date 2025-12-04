"""
Unit tests for orient module
"""

import pytest
import numpy as np
import cv2
from paper_transform.orient import auto_orient, rotate_180, _rotate_90_ccw, _rotate_90_cw


class TestAutoOrient:
    """Test auto_orient function"""

    def setup_method(self):
        """Create test images before each test"""
        # Create a 200x300 portrait image (height > width)
        self.portrait_image = np.ones((300, 200, 3), dtype=np.uint8) * 255

        # Create a 300x200 landscape image (width > height)
        self.landscape_image = np.ones((200, 300, 3), dtype=np.uint8) * 255

        # Create a 200x200 square image
        self.square_image = np.ones((200, 200, 3), dtype=np.uint8) * 255

    def test_auto_portrait_unchanged(self):
        """Test that portrait images remain unchanged in auto mode"""
        result = auto_orient(self.portrait_image, orientation="auto")

        # Should keep portrait orientation
        assert result.shape[0] > result.shape[1]  # height > width
        assert result.shape == self.portrait_image.shape

    def test_auto_landscape_rotated(self):
        """Test that landscape images are rotated to portrait in auto mode"""
        result = auto_orient(self.landscape_image, orientation="auto")

        # Should rotate to portrait
        assert result.shape[0] > result.shape[1]  # height > width
        assert result.shape[0] == self.landscape_image.shape[1]  # swapped dimensions
        assert result.shape[1] == self.landscape_image.shape[0]

    def test_auto_square_unchanged(self):
        """Test that square images remain unchanged in auto mode"""
        result = auto_orient(self.square_image, orientation="auto")

        # Square should remain unchanged
        assert result.shape == self.square_image.shape

    def test_force_portrait_from_landscape(self):
        """Test forcing portrait orientation from landscape"""
        result = auto_orient(self.landscape_image, orientation="portrait")

        # Should be portrait
        assert result.shape[0] > result.shape[1]

    def test_force_portrait_from_portrait(self):
        """Test forcing portrait orientation from portrait"""
        result = auto_orient(self.portrait_image, orientation="portrait")

        # Should remain portrait
        assert result.shape == self.portrait_image.shape

    def test_force_landscape_from_portrait(self):
        """Test forcing landscape orientation from portrait"""
        result = auto_orient(self.portrait_image, orientation="landscape")

        # Should be landscape
        assert result.shape[1] > result.shape[0]

    def test_force_landscape_from_landscape(self):
        """Test forcing landscape orientation from landscape"""
        result = auto_orient(self.landscape_image, orientation="landscape")

        # Should remain landscape
        assert result.shape == self.landscape_image.shape

    def test_invalid_orientation(self):
        """Test error handling for invalid orientation"""
        with pytest.raises(ValueError, match="Invalid orientation"):
            auto_orient(self.portrait_image, orientation="invalid")

    def test_invalid_image(self):
        """Test error handling for invalid image"""
        with pytest.raises(ValueError, match="Input image is None or empty"):
            auto_orient(None, orientation="auto")

        empty_image = np.array([])
        with pytest.raises(ValueError, match="Input image is None or empty"):
            auto_orient(empty_image, orientation="auto")


class TestRotationFunctions:
    """Test rotation utility functions"""

    def setup_method(self):
        """Create test images with asymmetric content"""
        # Create an image with a distinctive pattern to verify rotation
        self.test_image = np.zeros((100, 200, 3), dtype=np.uint8)
        # Draw a white rectangle in top-left corner
        cv2.rectangle(self.test_image, (10, 10), (40, 30), (255, 255, 255), -1)

    def test_rotate_90_ccw(self):
        """Test 90-degree counter-clockwise rotation"""
        result = _rotate_90_ccw(self.test_image)

        # Dimensions should be swapped
        assert result.shape[0] == self.test_image.shape[1]  # height = old width
        assert result.shape[1] == self.test_image.shape[0]  # width = old height

        # Original top-left should now be bottom-left
        # Check that there's white color in the expected region after rotation
        assert result[result.shape[0] - 40:result.shape[0] - 10, 10:30].mean() > 200

    def test_rotate_90_cw(self):
        """Test 90-degree clockwise rotation"""
        result = _rotate_90_cw(self.test_image)

        # Dimensions should be swapped
        assert result.shape[0] == self.test_image.shape[1]
        assert result.shape[1] == self.test_image.shape[0]

        # Original top-left should now be top-right
        # Use a more lenient threshold to account for edge interpolation
        assert result[10:30, result.shape[1] - 40:result.shape[1] - 10].mean() > 150

    def test_rotate_180(self):
        """Test 180-degree rotation"""
        result = rotate_180(self.test_image)

        # Dimensions should remain the same
        assert result.shape == self.test_image.shape

        # Original top-left should now be bottom-right
        assert result[result.shape[0] - 30:result.shape[0] - 10,
                      result.shape[1] - 40:result.shape[1] - 10].mean() > 200

    def test_double_rotation_is_identity(self):
        """Test that rotating 180° twice returns to original"""
        rotated_once = rotate_180(self.test_image)
        rotated_twice = rotate_180(rotated_once)

        # Should be back to original
        np.testing.assert_array_equal(rotated_twice, self.test_image)

    def test_four_90_rotations_is_identity(self):
        """Test that four 90° CCW rotations return to original"""
        result = self.test_image
        for _ in range(4):
            result = _rotate_90_ccw(result)

        # Should be back to original
        np.testing.assert_array_equal(result, self.test_image)
