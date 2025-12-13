"""
Unit tests for text orientation module
"""

import pytest
import numpy as np
import cv2
from paper_transform.document_orient.orient_text import (
    _rotate_image,
    _calculate_text_score,
    orient_by_text,
    _fallback_to_geometric
)


class TestRotateImage:
    """Test image rotation function"""

    def setup_method(self):
        """Create test image"""
        # Create 100x200 test image (portrait)
        self.test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.rectangle(self.test_image, (10, 10), (30, 30), (0, 0, 0), -1)

    def test_rotate_0(self):
        """Test 0-degree rotation (no change)"""
        result = _rotate_image(self.test_image, 0)
        np.testing.assert_array_equal(result, self.test_image)

    def test_rotate_90(self):
        """Test 90-degree rotation"""
        result = _rotate_image(self.test_image, 90)
        # Dimensions should be swapped
        assert result.shape[0] == self.test_image.shape[1]
        assert result.shape[1] == self.test_image.shape[0]

    def test_rotate_180(self):
        """Test 180-degree rotation"""
        result = _rotate_image(self.test_image, 180)
        # Dimensions stay the same
        assert result.shape == self.test_image.shape

    def test_rotate_270(self):
        """Test 270-degree rotation"""
        result = _rotate_image(self.test_image, 270)
        # Dimensions should be swapped
        assert result.shape[0] == self.test_image.shape[1]
        assert result.shape[1] == self.test_image.shape[0]

    def test_invalid_angle(self):
        """Test error handling for invalid angle"""
        with pytest.raises(ValueError, match="Invalid rotation angle"):
            _rotate_image(self.test_image, 45)


class TestCalculateTextScore:
    """Test text scoring function"""

    def test_empty_text(self):
        """Test score for empty text"""
        score = _calculate_text_score("")
        assert score == 0.0

    def test_whitespace_only(self):
        """Test score for whitespace-only text"""
        score = _calculate_text_score("   \n\t  ")
        assert score == 0.0

    def test_good_text(self):
        """Test score for good quality text"""
        text = "This is a well-formed sentence with proper text."
        score = _calculate_text_score(text)
        # Should have high score (length * high purity)
        assert score > 40.0

    def test_text_with_punctuation(self):
        """Test that punctuation is considered 'good'"""
        text = "Hello, world! How are you?"
        score = _calculate_text_score(text)
        # Should have good score
        assert score > 20.0

    def test_garbage_text(self):
        """Test score for garbage text (low purity)"""
        text = "###@@@$$$%%%^^^&&&***"
        score = _calculate_text_score(text)
        # Should have very low score
        assert score < 5.0

    def test_mixed_quality(self):
        """Test that more readable text scores higher"""
        good_text = "The quick brown fox jumps over the lazy dog."
        bad_text = "Th3 qu!ck br0wn f0x jump$ 0v3r th3 l@zy d0g."

        good_score = _calculate_text_score(good_text)
        bad_score = _calculate_text_score(bad_text)

        # Good text should score higher
        assert good_score > bad_score

    def test_diacritics_allowed(self):
        """Test that diacritics (Czech, etc.) are considered good"""
        text = "Příliš žluťoučký kůň úpěl ďábelské ódy."
        score = _calculate_text_score(text)
        # Should have reasonable score (diacritics are allowed)
        assert score > 30.0


class TestOrientByText:
    """Test orient_by_text function"""

    def setup_method(self):
        """Create test images"""
        # Create a simple 200x300 portrait image
        self.portrait_image = np.ones((300, 200, 3), dtype=np.uint8) * 255

    def test_empty_image_raises_error(self):
        """Test that empty image raises ValueError"""
        with pytest.raises(ValueError, match="Input image is None or empty"):
            orient_by_text(None)

        empty_img = np.array([])
        with pytest.raises(ValueError, match="Input image is None or empty"):
            orient_by_text(empty_img)

    def test_invalid_ocr_engine_raises_error(self):
        """Test that invalid OCR engine raises ValueError"""
        with pytest.raises(ValueError, match="Unknown OCR engine"):
            orient_by_text(self.portrait_image, ocr_engine="invalid")

    def test_returns_image_with_same_channels(self):
        """Test that output has same number of channels as input"""
        # This test will use fallback if OCR is not available
        result = orient_by_text(self.portrait_image, min_score_threshold=1000.0)

        # Should return BGR image with 3 channels
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_returns_valid_dimensions(self):
        """Test that output has valid dimensions"""
        result = orient_by_text(self.portrait_image, min_score_threshold=1000.0)

        # Should have positive dimensions
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    # Note: Full OCR tests require Tesseract installation
    # These would be integration tests rather than unit tests


class TestFallbackToGeometric:
    """Test geometric fallback function"""

    def test_fallback_returns_image(self):
        """Test that fallback returns a valid image"""
        test_image = np.ones((200, 300, 3), dtype=np.uint8) * 255

        result = _fallback_to_geometric(test_image)

        # Should return an image with same number of channels
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_fallback_preserves_or_rotates(self):
        """Test that fallback either preserves or rotates to portrait"""
        # Landscape image
        landscape = np.ones((200, 300, 3), dtype=np.uint8) * 255

        result = _fallback_to_geometric(landscape)

        # Should either keep landscape or rotate to portrait
        # We just verify we got a valid image back
        assert result.shape[0] > 0
        assert result.shape[1] > 0
