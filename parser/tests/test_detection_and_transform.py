"""
End-to-end tests for paper detection and transformation pipeline

These tests demonstrate the complete workflow:
1. Detect paper corners in image
2. Apply perspective transformation to extract paper
3. Save visualization of results
"""

import pytest
import cv2
import numpy as np
import json
from pathlib import Path

from paper_detection import PaperDetector
from paper_transform import warp_paper, orient_by_text


class TestDetectionAndTransform:
    """Test complete detection and transformation pipeline"""

    @pytest.fixture
    def detector(self):
        """Create paper detector instance"""
        try:
            return PaperDetector()
        except FileNotFoundError:
            pytest.skip("Paper detection model not trained yet")

    @pytest.fixture
    def output_dir(self):
        """Output directory for test results"""
        output_dir = Path("parser/tests/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @pytest.fixture
    def test_images(self):
        """Get list of test images"""
        images_dir = Path("paper_detection/data/images")
        corners_file = Path("paper_detection/data/corners.json")

        if not images_dir.exists() or not corners_file.exists():
            pytest.skip("Test data not available")

        # Load corners to get image list
        with open(corners_file, 'r') as f:
            corners_data = json.load(f)

        # Get all images that have annotations
        test_images = []
        for image_name in corners_data.keys():
            image_path = images_dir / image_name
            if image_path.exists():
                test_images.append(image_path)

        return test_images

    def test_detect_and_transform_all(self, detector, test_images, output_dir):
        """
        Test detection and transformation on all test images

        For each image:
        1. Detect paper corners
        2. Transform perspective to extract paper
        3. Save side-by-side comparison
        """
        if not test_images:
            pytest.skip("No test images available")

        print(f"\nProcessing {len(test_images)} test images...")
        print(f"Output directory: {output_dir}")

        success_count = 0
        failed_images = []

        for image_path in test_images:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  ⚠ Failed to load: {image_path.name}")
                continue

            # Detect paper corners
            corners = detector.detect(image, debug=False)

            if corners is None:
                print(f"  ✗ {image_path.name} - Detection failed")
                failed_images.append(image_path.name)
                continue

            # Transform paper
            try:
                warped = warp_paper(image, corners)

                # Orient based on text
                try:
                    oriented = orient_by_text(warped, debug=False)
                except Exception:
                    # Fallback to warped without orientation if OCR fails
                    oriented = warped

                # Save visualization
                self._save_result(image, corners, warped, oriented, image_path.name, output_dir)

                print(f"  ✓ {image_path.name}")
                success_count += 1

            except Exception as e:
                print(f"  ✗ {image_path.name} - Transform failed: {e}")
                failed_images.append(image_path.name)

        # Print summary
        print("\n" + "="*60)
        print("Pipeline Test Summary")
        print("="*60)
        print(f"Total images: {len(test_images)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(failed_images)}")
        print(f"Success rate: {success_count/len(test_images)*100:.1f}%")

        if failed_images:
            print(f"\nFailed images:")
            for name in failed_images:
                print(f"  - {name}")

        print("="*60)

    def test_detect_and_transform_sample(self, detector, test_images, output_dir):
        """
        Test detection and transformation on first 5 images

        Creates detailed visualization for quick verification
        """
        if not test_images:
            pytest.skip("No test images available")

        sample_images = test_images[:5]
        print(f"\nProcessing sample of {len(sample_images)} images...")

        for image_path in sample_images:
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            # Detect corners
            corners = detector.detect(image, debug=True)

            if corners is not None:
                # Transform
                warped = warp_paper(image, corners)

                # Orient based on text
                try:
                    oriented = orient_by_text(warped, debug=True)
                except Exception:
                    oriented = warped

                # Save detailed visualization
                self._save_detailed_result(image, corners, warped, oriented, image_path.name, output_dir)
                print(f"  ✓ Saved detailed visualization for {image_path.name}")

    def test_pipeline_statistics(self, detector, test_images):
        """
        Collect statistics about detection and transformation pipeline
        """
        if not test_images:
            pytest.skip("No test images available")

        # Limit to first 20 for speed
        sample = test_images[:20]

        stats = {
            'detected': 0,
            'transformed': 0,
            'failed_detection': 0,
            'failed_transform': 0,
            'output_sizes': []
        }

        for image_path in sample:
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            # Try detection
            corners = detector.detect(image, debug=False)

            if corners is None:
                stats['failed_detection'] += 1
                continue

            stats['detected'] += 1

            # Try transformation
            try:
                warped = warp_paper(image, corners)
                stats['transformed'] += 1
                stats['output_sizes'].append(warped.shape[:2])
            except Exception:
                stats['failed_transform'] += 1

        # Print statistics
        print("\n" + "="*60)
        print("Pipeline Statistics")
        print("="*60)
        print(f"Images processed: {len(sample)}")
        print(f"\nDetection:")
        print(f"  Success: {stats['detected']}")
        print(f"  Failed: {stats['failed_detection']}")
        print(f"  Rate: {stats['detected']/len(sample)*100:.1f}%")
        print(f"\nTransformation:")
        print(f"  Success: {stats['transformed']}")
        print(f"  Failed: {stats['failed_transform']}")

        if stats['output_sizes']:
            heights = [s[0] for s in stats['output_sizes']]
            widths = [s[1] for s in stats['output_sizes']]
            print(f"\nOutput sizes:")
            print(f"  Height: {min(heights)} - {max(heights)} px (avg: {sum(heights)/len(heights):.0f})")
            print(f"  Width: {min(widths)} - {max(widths)} px (avg: {sum(widths)/len(widths):.0f})")

        print("="*60)

    def _save_result(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        warped: np.ndarray,
        oriented: np.ndarray,
        image_name: str,
        output_dir: Path
    ):
        """
        Save side-by-side comparison of original, warped and oriented

        Layout: Original with corners | Warped | Oriented
        """
        # Draw corners on original
        original_vis = image.copy()
        corners_int = corners.astype(np.int32)

        # Draw polygon
        cv2.polylines(original_vis, [corners_int], True, (0, 255, 0), 3)

        # Draw corner points
        labels = ['TL', 'TR', 'BR', 'BL']
        for idx, corner in enumerate(corners_int):
            cv2.circle(original_vis, tuple(corner), 8, (0, 255, 0), -1)
            cv2.putText(
                original_vis, labels[idx],
                tuple(corner + np.array([10, -10])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # Create 3-column layout
        h1, w1 = original_vis.shape[:2]
        h2, w2 = warped.shape[:2]
        h3, w3 = oriented.shape[:2]

        # Scale all to match original height
        scale2 = h1 / h2
        new_w2 = int(w2 * scale2)
        warped_scaled = cv2.resize(warped, (new_w2, h1), interpolation=cv2.INTER_LINEAR)

        scale3 = h1 / h3
        new_w3 = int(w3 * scale3)
        oriented_scaled = cv2.resize(oriented, (new_w3, h1), interpolation=cv2.INTER_LINEAR)

        # Create canvas with 3 columns
        total_width = w1 + new_w2 + new_w3 + 40  # 20px gaps between columns
        canvas = np.zeros((h1, total_width, 3), dtype=np.uint8)

        canvas[:, :w1] = original_vis
        canvas[:, w1+20:w1+20+new_w2] = warped_scaled
        canvas[:, w1+new_w2+40:w1+new_w2+40+new_w3] = oriented_scaled

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Original + Corners", (10, 40), font, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, "Warped", (w1 + 30, 40), font, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, "Oriented (Text)", (w1 + new_w2 + 50, 40), font, 1.0, (255, 255, 255), 2)

        # Save
        output_path = output_dir / f"pipeline_{image_name}"
        cv2.imwrite(str(output_path), canvas)

    def _save_detailed_result(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        warped: np.ndarray,
        oriented: np.ndarray,
        image_name: str,
        output_dir: Path
    ):
        """
        Save detailed 4-column visualization

        Layout: Original | Original with corners | Warped | Oriented
        """
        # Original
        original = image.copy()

        # Original with corners
        original_corners = image.copy()
        corners_int = corners.astype(np.int32)
        cv2.polylines(original_corners, [corners_int], True, (0, 255, 0), 3)

        labels = ['TL', 'TR', 'BR', 'BL']
        for idx, corner in enumerate(corners_int):
            cv2.circle(original_corners, tuple(corner), 8, (0, 255, 0), -1)
            cv2.putText(
                original_corners, labels[idx],
                tuple(corner + np.array([10, -10])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # Scale all to same height
        h, w = original.shape[:2]
        h2, w2 = warped.shape[:2]
        h3, w3 = oriented.shape[:2]

        scale2 = h / h2
        new_w2 = int(w2 * scale2)
        warped_scaled = cv2.resize(warped, (new_w2, h), interpolation=cv2.INTER_LINEAR)

        scale3 = h / h3
        new_w3 = int(w3 * scale3)
        oriented_scaled = cv2.resize(oriented, (new_w3, h), interpolation=cv2.INTER_LINEAR)

        # Create 4-column canvas with gaps
        total_width = w * 2 + new_w2 + new_w3 + 60  # 20px gaps between columns
        canvas = np.zeros((h, total_width, 3), dtype=np.uint8)

        # Place images with gaps
        canvas[:, :w] = original
        canvas[:, w+20:2*w+20] = original_corners
        canvas[:, 2*w+40:2*w+40+new_w2] = warped_scaled
        canvas[:, 2*w+new_w2+60:2*w+new_w2+60+new_w3] = oriented_scaled

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Original", (10, 40), font, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, "Detected Corners", (w + 30, 40), font, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, "Warped", (2*w + 50, 40), font, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, "Oriented (Text)", (2*w + new_w2 + 70, 40), font, 1.0, (255, 255, 255), 2)

        # Save
        output_path = output_dir / f"detailed_{image_name}"
        cv2.imwrite(str(output_path), canvas)
