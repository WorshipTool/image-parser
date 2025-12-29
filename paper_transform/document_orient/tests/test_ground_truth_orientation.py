"""
Ground truth orientation tests

These tests validate text-based orientation against known correct orientations.
Test images have been manually verified for expected rotation angles.
"""

import pytest
import cv2
import numpy as np
import json
from pathlib import Path

from paper_transform.document_orient import orient_by_text


class TestGroundTruthOrientation:
    """Test orientation against manually annotated ground truth"""

    @pytest.fixture
    def test_data_dir(self):
        """Get test data directory"""
        return Path(__file__).parent / "test_data"

    @pytest.fixture
    def ground_truth(self, test_data_dir):
        """Load ground truth annotations"""
        ground_truth_file = test_data_dir / "ground_truth.json"

        if not ground_truth_file.exists():
            pytest.skip("Ground truth file not found. Run temp/create_orientation_test_dataset.py first")

        with open(ground_truth_file, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def test_images(self, test_data_dir, ground_truth):
        """Load test images with ground truth"""
        images = {}

        for image_name, annotations in ground_truth.items():
            image_path = test_data_dir / image_name

            if not image_path.exists():
                continue

            image = cv2.imread(str(image_path))
            if image is not None:
                images[image_name] = {
                    "image": image,
                    "expected_rotation": annotations["expected_rotation"],
                    "description": annotations.get("description", ""),
                    "source": annotations.get("source", "")
                }

        if not images:
            pytest.skip("No test images found")

        return images

    def _apply_rotation(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Apply rotation to image"""
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            raise ValueError(f"Invalid rotation angle: {angle}")

    def _detect_applied_rotation(self, original: np.ndarray, oriented: np.ndarray) -> int:
        """
        Detect what rotation was applied by comparing shapes and content.

        Returns:
            Rotation angle (0, 90, 180, 270)
        """
        orig_h, orig_w = original.shape[:2]
        new_h, new_w = oriented.shape[:2]

        # Check shape to narrow down options
        if orig_h == new_h and orig_w == new_w:
            # Either 0° or 180°
            # Compare top-left corners to distinguish
            # If top-left corner is very different, it's 180°
            orig_corner = original[:50, :50]
            new_corner = oriented[:50, :50]
            diff = np.mean(np.abs(orig_corner.astype(float) - new_corner.astype(float)))

            if diff < 10:  # Very similar
                return 0
            else:
                return 180

        elif orig_h == new_w and orig_w == new_h:
            # Either 90° or 270°
            # Compare if original top-left matches new top-right (90° CCW)
            # or original top-left matches new bottom-left (270° CCW = 90° CW)
            orig_tl = original[:50, :50]
            new_tr = oriented[:50, -50:]
            new_bl = oriented[-50:, :50]

            diff_tr = np.mean(np.abs(orig_tl.astype(float) - cv2.resize(new_tr, (50, 50)).astype(float)))
            diff_bl = np.mean(np.abs(orig_tl.astype(float) - cv2.resize(new_bl, (50, 50)).astype(float)))

            if diff_tr < diff_bl:
                return 90
            else:
                return 270
        else:
            # Shouldn't happen with valid rotations
            return 0

    def test_orientation_accuracy_with_ocr(self, test_images):
        """
        Test orientation accuracy with OCR (if available)

        This test checks if the orientation algorithm correctly identifies
        the rotation needed to make documents readable.
        """
        try:
            import pytesseract
            # Check if Tesseract is actually available
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                pytest.skip("Tesseract OCR not installed")
        except ImportError:
            pytest.skip("pytesseract not installed")

        results = []
        correct = 0
        total = 0

        print("\n" + "=" * 70)
        print("Testing Orientation with OCR")
        print("=" * 70)

        for image_name, data in test_images.items():
            # Image is stored in its "wrong" orientation
            # expected_rotation tells us what rotation is needed to fix it
            image = data["image"]
            expected_rotation = data["expected_rotation"]
            description = data["description"]

            # Try to orient the image AS IS
            try:
                oriented = orient_by_text(image, debug=False)

                # Detect what rotation was applied
                applied_rotation = self._detect_applied_rotation(image, oriented)

                # Check if it matches expected
                is_correct = (applied_rotation == expected_rotation)

                results.append({
                    "image": image_name,
                    "expected": expected_rotation,
                    "detected": applied_rotation,
                    "correct": is_correct
                })

                status = "✓" if is_correct else "✗"
                print(f"\n{status} {image_name}")
                print(f"  Description: {description}")
                print(f"  Expected rotation: {expected_rotation}°")
                print(f"  Applied rotation: {applied_rotation}°")

                if is_correct:
                    correct += 1
                total += 1

            except Exception as e:
                print(f"\n✗ {image_name} - Error: {e}")
                results.append({
                    "image": image_name,
                    "expected": expected_rotation,
                    "detected": None,
                    "correct": False,
                    "error": str(e)
                })
                total += 1

        print("\n" + "=" * 70)
        print(f"Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
        print("=" * 70)

        # We expect at least 90% accuracy with OCR (high threshold to catch errors)
        accuracy = correct / total if total > 0 else 0
        assert accuracy >= 0.9, f"Orientation accuracy too low: {accuracy:.1%}"

    def test_orientation_with_geometric_fallback(self, test_images):
        """
        Test orientation with geometric fallback (no OCR required)

        This test ensures the system works even without OCR by using
        aspect ratio heuristics.
        """
        results = []
        portrait_correct = 0
        total = 0

        print("\n" + "=" * 70)
        print("Testing Orientation with Geometric Fallback")
        print("=" * 70)

        for image_name, data in test_images.items():
            image = data["image"]
            expected_rotation = data["expected_rotation"]
            description = data["description"]

            # Apply inverse rotation
            inverse_rotation = (360 - expected_rotation) % 360
            misoriented = self._apply_rotation(image, inverse_rotation)

            # Try to orient (will fall back to geometric)
            try:
                oriented = orient_by_text(misoriented, debug=False)

                # For geometric orientation, we mainly check if it's portrait
                h, w = oriented.shape[:2]
                is_portrait = h > w

                print(f"\n  {image_name}")
                print(f"    Misoriented shape: {misoriented.shape[1]}x{misoriented.shape[0]}")
                print(f"    Oriented shape: {w}x{h}")
                print(f"    Is portrait: {is_portrait}")

                # Count as correct if result is portrait (most documents are)
                if is_portrait:
                    portrait_correct += 1
                total += 1

            except Exception as e:
                print(f"\n  {image_name} - Error: {e}")
                total += 1

        print("\n" + "=" * 70)
        print(f"Portrait orientation: {portrait_correct}/{total} ({portrait_correct/total*100:.1f}%)")
        print("=" * 70)

        # Geometric fallback should make most images portrait
        rate = portrait_correct / total if total > 0 else 0
        assert rate >= 0.6, f"Too few images oriented to portrait: {rate:.1%}"

    def test_individual_upside_down_documents(self, test_images):
        """
        Test specific cases of upside down documents

        This test specifically validates that upside down documents
        are correctly identified and rotated 180 degrees.
        """
        upside_down_images = {
            name: data
            for name, data in test_images.items()
            if data["expected_rotation"] == 180
        }

        if not upside_down_images:
            pytest.skip("No upside down test images available")

        # Skip if OCR not available
        try:
            import pytesseract
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                pytest.skip("Tesseract OCR not installed - needed for 180° detection")
        except ImportError:
            pytest.skip("pytesseract not installed - needed for 180° detection")

        print("\n" + "=" * 70)
        print("Testing Upside Down Document Detection")
        print("=" * 70)

        correct = 0
        total = 0

        for image_name, data in upside_down_images.items():
            # Image is already upside down (as stored in test data)
            # We expect orient_by_text to detect this and rotate it 180°
            image = data["image"]

            # Try to orient
            oriented = orient_by_text(image, debug=False)

            # Check if 180° rotation was applied
            applied_rotation = self._detect_applied_rotation(image, oriented)
            is_correct = (applied_rotation == 180)

            status = "✓" if is_correct else "✗"
            print(f"\n{status} {image_name}")
            print(f"  Applied rotation: {applied_rotation}° (expected: 180°)")

            if is_correct:
                correct += 1
            total += 1

        print("\n" + "=" * 70)
        print(f"Upside down detection: {correct}/{total} correct ({correct/total*100:.1f}%)")
        print("=" * 70)

        # Should detect at least half of upside down documents
        accuracy = correct / total if total > 0 else 0
        assert accuracy >= 0.5, f"Upside down detection too low: {accuracy:.1%}"

    def test_correctly_oriented_documents_unchanged(self, test_images):
        """
        Test that correctly oriented documents remain unchanged

        Documents that are already correctly oriented should not be rotated.
        """
        correct_images = {
            name: data
            for name, data in test_images.items()
            if data["expected_rotation"] == 0
        }

        if not correct_images:
            pytest.skip("No correctly oriented test images available")

        print("\n" + "=" * 70)
        print("Testing Correctly Oriented Documents")
        print("=" * 70)

        unchanged = 0
        total = 0

        for image_name, data in correct_images.items():
            image = data["image"]

            # Orient (should remain the same)
            oriented = orient_by_text(image, debug=False)

            # Check if no rotation was applied
            applied_rotation = self._detect_applied_rotation(image, oriented)
            is_unchanged = (applied_rotation == 0)

            status = "✓" if is_unchanged else "✗"
            print(f"\n{status} {image_name}")
            print(f"  Applied rotation: {applied_rotation}° (expected: 0°)")

            if is_unchanged:
                unchanged += 1
            total += 1

        print("\n" + "=" * 70)
        print(f"Unchanged: {unchanged}/{total} ({unchanged/total*100:.1f}%)")
        print("=" * 70)

        # Most correctly oriented docs should stay unchanged
        rate = unchanged / total if total > 0 else 0
        assert rate >= 0.5, f"Too many correctly oriented docs were rotated: {rate:.1%}"
