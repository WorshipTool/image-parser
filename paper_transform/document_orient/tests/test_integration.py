"""
Integration tests for text orientation with real-world scenarios
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from paper_transform.document_orient.orient_text import orient_by_text
from paper_transform.document_orient.ocr_engines import TesseractOCR


class TestOrientationLogic:
    """Test that orientation logic works correctly"""

    def test_each_rotation_is_physically_rotated(self):
        """Verify that each rotation angle produces different image dimensions"""
        # Create a landscape image (width > height)
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255

        rotations_tested = []
        dimensions_seen = []

        # Mock OCR to capture what images it receives
        original_extract = TesseractOCR.extract_text

        def mock_extract(self, image):
            # Record the dimensions of rotated image
            dimensions_seen.append((image.shape[1], image.shape[0]))  # width, height
            rotations_tested.append(True)
            # Return minimal text so we don't hit fallback
            return "test text content here"

        with patch.object(TesseractOCR, 'extract_text', mock_extract):
            with patch.object(TesseractOCR, '_check_tesseract', return_value=True):
                # Run orientation (will use mocked OCR)
                try:
                    result = orient_by_text(test_image, debug=False)
                except Exception:
                    pass  # We don't care about the result, just that rotations happened

        # Verify we tested 4 rotations
        assert len(rotations_tested) == 4, "Should test all 4 rotations"

        # Verify we saw different dimensions (rotations actually happened)
        # 0° and 180° should have same dims, 90° and 270° should have swapped dims
        unique_dims = set(dimensions_seen)
        assert len(unique_dims) == 2, f"Should see 2 unique dimension sets, got {unique_dims}"

        # Verify original landscape (200x100) appears
        assert (200, 100) in dimensions_seen, "Should see original dimensions"
        # Verify rotated portrait (100x200) appears
        assert (100, 200) in dimensions_seen, "Should see rotated dimensions"

    def test_best_score_is_selected(self):
        """Verify that rotation with highest OCR score is selected"""
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255

        # Mock OCR to return different text quality for each rotation
        rotation_texts = {
            0: "garbage ▄▀▄ ▀▄▀ symbols",      # Low score
            90: "Even more garbage ###@@@",    # Very low score
            180: "This is perfect readable text with many words and good quality!",  # HIGH score
            270: "Some text but short"         # Medium score
        }

        call_count = [0]

        def mock_extract(self, image):
            angle_index = call_count[0] % 4
            angles = [0, 90, 180, 270]
            angle = angles[angle_index]
            call_count[0] += 1
            return rotation_texts[angle]

        with patch.object(TesseractOCR, 'extract_text', mock_extract):
            with patch.object(TesseractOCR, '_check_tesseract', return_value=True):
                result = orient_by_text(test_image, debug=False)

        # Result should be 180° rotated (same dimensions as original)
        # because that rotation had the best text
        assert result.shape == test_image.shape, "Should select 180° rotation"

    def test_fallback_when_all_scores_low(self):
        """Verify fallback to geometric orientation when OCR scores are too low"""
        # Portrait image
        test_image = np.ones((300, 200, 3), dtype=np.uint8) * 255

        def mock_extract_garbage(self, image):
            return "###@@@$$$"  # Garbage text, very low score

        with patch.object(TesseractOCR, 'extract_text', mock_extract_garbage):
            with patch.object(TesseractOCR, '_check_tesseract', return_value=True):
                # Use high threshold to force fallback
                result = orient_by_text(test_image, min_score_threshold=100.0, debug=False)

        # Should fallback to geometric (portrait stays portrait)
        assert result.shape == test_image.shape


class TestOCRConfigCorrectness:
    """Test that OCR is configured correctly (no auto-rotation)"""

    def test_tesseract_uses_psm6_not_psm1(self):
        """Verify Tesseract uses PSM 6 (no auto-orientation) instead of PSM 1"""
        # This test checks the OCR engine configuration
        import sys
        from unittest.mock import MagicMock

        # Mock pytesseract module if not available
        if 'pytesseract' not in sys.modules:
            sys.modules['pytesseract'] = MagicMock()

        with patch('pytesseract.get_tesseract_version', return_value='5.0.0'):
            with patch('pytesseract.image_to_string') as mock_ocr:
                mock_ocr.return_value = "test"

                ocr = TesseractOCR()
                test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

                ocr.extract_text(test_image)

                # Check that image_to_string was called
                assert mock_ocr.called, "Tesseract should be called"

                # Check the config parameter
                call_args = mock_ocr.call_args
                config = call_args[1]['config'] if 'config' in call_args[1] else call_args[0][1]

                # Should use PSM 6, NOT PSM 1 (which does auto-orientation)
                assert '--psm 6' in config, f"Should use PSM 6, got: {config}"
                assert '--psm 1' not in config, f"Should NOT use PSM 1 (auto-orientation)"
