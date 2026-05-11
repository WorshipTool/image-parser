"""
Test shouldCrop decision mechanism

Tests the paper detector's ability to distinguish between:
- Screenshots (shouldCrop = False)
- Real photos of paper (shouldCrop = True)
"""

import cv2
import pytest
from pathlib import Path
from paper_detection import PaperDetector


class TestShouldCropDecision:
    """Test shouldCrop decision for screenshots vs photos"""

    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return PaperDetector()

    @pytest.fixture
    def screenshots_dir(self):
        """Path to test screenshots directory"""
        return Path(__file__).parent / "test_data" / "screenshots"

    @pytest.fixture
    def photos_dir(self):
        """Path to test photos directory"""
        return Path(__file__).parent.parent / "data" / "images"

    @pytest.fixture
    def screenshot_images(self, screenshots_dir):
        """List of screenshot filenames that should be rejected"""
        return [
            "chvalykazaniczchvalychvalyasp?cekam-na-tebehtm.png",
            "kytaristkaczzpevnikluciesen.png",
            "akordikyczcesta.png"
        ]

    @pytest.fixture
    def photo_images(self):
        """List of photo filenames that should be accepted"""
        return [
            "IMG_20230826_092429.jpg",
            "IMG_20230826_093159.jpg",
            "IMG_20230826_092437.jpg"
        ]

    @pytest.mark.parametrize("screenshot_name", [
        "chvalykazaniczchvalychvalyasp?cekam-na-tebehtm.png",
        "kytaristkaczzpevnikluciesen.png",
        "akordikyczcesta.png"
    ])
    def test_screenshots_rejected(self, detector, screenshots_dir, screenshot_name):
        """Test that screenshots are correctly rejected (shouldCrop = False)"""
        image_path = screenshots_dir / screenshot_name
        assert image_path.exists(), f"Screenshot not found: {image_path}"

        # Load image
        image = cv2.imread(str(image_path))
        assert image is not None, f"Failed to load image: {image_path}"

        # Run detection
        result = detector.detect(image, debug=True)

        # Verify screenshot is rejected
        assert result['shouldCrop'] is False, (
            f"Screenshot {screenshot_name} should be rejected (shouldCrop = False)\n"
            f"Got: shouldCrop = {result['shouldCrop']}\n"
            f"Rejection reason: {result['debug']['heatmap_analysis'].get('rejection_reason', 'None')}"
        )

        # Corners should be None
        assert result['corners'] is None, (
            f"Screenshot {screenshot_name} should have corners = None"
        )

        # Print debug info
        heatmap = result['debug']['heatmap_analysis']
        print(f"\n✓ {screenshot_name} correctly rejected")
        print(f"  Reason: {heatmap['rejection_reason']}")
        print(f"  area_ratio: {heatmap['metrics']['area_ratio']:.3f}")
        print(f"  bbox_cover: {heatmap['metrics']['bbox_cover']:.3f}")

    @pytest.mark.parametrize("photo_name", [
        "IMG_20230826_092429.jpg",
        "IMG_20230826_093159.jpg",
        "IMG_20230826_092437.jpg"
    ])
    def test_photos_accepted(self, detector, photos_dir, photo_name):
        """Test that real photos are correctly accepted (shouldCrop = True)"""
        image_path = photos_dir / photo_name
        assert image_path.exists(), f"Photo not found: {image_path}"

        # Load image
        image = cv2.imread(str(image_path))
        assert image is not None, f"Failed to load image: {image_path}"

        # Run detection
        result = detector.detect(image, debug=True)

        # Verify photo is accepted
        assert result['shouldCrop'] is True, (
            f"Photo {photo_name} should be accepted (shouldCrop = True)\n"
            f"Got: shouldCrop = {result['shouldCrop']}\n"
            f"Debug info: {result['debug']}"
        )

        # Corners should be found
        assert result['corners'] is not None, (
            f"Photo {photo_name} should have corners detected"
        )

        # Print debug info
        heatmap = result['debug']['heatmap_analysis']
        print(f"\n✓ {photo_name} correctly accepted")
        print(f"  area_ratio: {heatmap['metrics']['area_ratio']:.3f}")
        print(f"  bbox_cover: {heatmap['metrics']['bbox_cover']:.3f}")

    def test_summary(self, detector, screenshots_dir, photos_dir, screenshot_images, photo_images):
        """Run all images and print summary"""
        results = []

        # Test screenshots
        for screenshot_name in screenshot_images:
            image_path = screenshots_dir / screenshot_name
            if not image_path.exists():
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            result = detector.detect(image, debug=False)
            results.append({
                'name': screenshot_name,
                'type': 'screenshot',
                'expected': False,
                'actual': result['shouldCrop'],
                'correct': result['shouldCrop'] is False
            })

        # Test photos
        for photo_name in photo_images:
            image_path = photos_dir / photo_name
            if not image_path.exists():
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            result = detector.detect(image, debug=False)
            results.append({
                'name': photo_name,
                'type': 'photo',
                'expected': True,
                'actual': result['shouldCrop'],
                'correct': result['shouldCrop'] is True
            })

        # Print summary
        print("\n" + "="*70)
        print("SHOULD_CROP DECISION TEST SUMMARY")
        print("="*70)

        screenshots_tested = [r for r in results if r['type'] == 'screenshot']
        photos_tested = [r for r in results if r['type'] == 'photo']

        print(f"\nScreenshots (expected shouldCrop = False):")
        for r in screenshots_tested:
            status = "✅" if r['correct'] else "❌"
            print(f"  {status} {r['name']}: {r['actual']}")

        print(f"\nPhotos (expected shouldCrop = True):")
        for r in photos_tested:
            status = "✅" if r['correct'] else "❌"
            print(f"  {status} {r['name']}: {r['actual']}")

        # Calculate accuracy
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        print(f"\nOverall: {correct}/{total} correct ({correct/total*100:.0f}%)")
        print("="*70)

        # Fail if any incorrect
        assert all(r['correct'] for r in results), "Some images were incorrectly classified"
