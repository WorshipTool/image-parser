"""
Test for model visualization - generates heatmaps for all test images
"""

import pytest
import cv2
import numpy as np
import json
from pathlib import Path

from paper_detection.model.infer import SegmentationInference
from paper_detection.model.config import ModelConfig


class TestModelVisualization:
    """Test model output visualization"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return ModelConfig()

    @pytest.fixture
    def inference(self, config):
        """Create inference engine"""
        try:
            return SegmentationInference(config=config)
        except FileNotFoundError:
            pytest.skip("Model not trained yet")

    @pytest.fixture
    def output_dir(self):
        """Output directory for visualizations"""
        output_dir = Path("paper_detection/model/tests/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @pytest.fixture
    def test_images(self, config):
        """Get list of test images"""
        images_dir = Path(config.IMAGES_DIR)
        corners_file = Path(config.CORNERS_FILE)

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

    def test_generate_heatmaps(self, inference, test_images, output_dir):
        """
        Generate heatmap visualizations for all test images

        This test creates:
        - Probability heatmap (colormap showing mask confidence)
        - Binary mask overlay
        - Detected corners overlay
        """
        if not test_images:
            pytest.skip("No test images available")

        print(f"\nGenerating heatmaps for {len(test_images)} images...")
        print(f"Output directory: {output_dir}")

        for image_path in test_images:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  ⚠ Failed to load: {image_path.name}")
                continue

            # Get predictions
            mask_prob, mask_resized = inference.predict_mask(image)
            corners = inference.detect_corners(image, debug=False)

            # Create visualizations
            self._save_heatmap(image, mask_prob, mask_resized, corners, image_path.name, output_dir)

            print(f"  ✓ {image_path.name}")

    def _save_heatmap(
        self,
        image: np.ndarray,
        mask_prob: np.ndarray,
        mask_resized: np.ndarray,
        corners: np.ndarray,
        image_name: str,
        output_dir: Path
    ):
        """
        Save heatmap visualization

        Creates a 3-column visualization:
        - Column 1: Original image
        - Column 2: Probability heatmap
        - Column 3: Detected corners + mask overlay
        """
        h, w = image.shape[:2]

        # Create output canvas (3 columns)
        canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)

        # Column 1: Original image
        canvas[:, :w] = image

        # Column 2: Probability heatmap
        # Convert probability mask to heatmap
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)

        # Blend with original image for context
        heatmap_overlay = cv2.addWeighted(image, 0.4, heatmap, 0.6, 0)
        canvas[:, w:2*w] = heatmap_overlay

        # Column 3: Corners + mask overlay
        result = image.copy()

        # Overlay mask
        mask_colored = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        result = cv2.addWeighted(result, 0.6, mask_colored, 0.4, 0)

        # Draw corners if detected
        if corners is not None:
            corners_int = corners.astype(np.int32)

            # Draw polygon
            cv2.polylines(result, [corners_int], True, (0, 255, 0), 3)

            # Draw corner points with labels
            labels = ['TL', 'TR', 'BR', 'BL']
            for idx, corner in enumerate(corners_int):
                cv2.circle(result, tuple(corner), 8, (0, 255, 0), -1)
                cv2.putText(
                    result, labels[idx],
                    tuple(corner + np.array([10, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

        canvas[:, 2*w:] = result

        # Add column labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)

        cv2.putText(canvas, "Original", (10, 40), font, font_scale, text_color, font_thickness)
        cv2.putText(canvas, "Probability Heatmap", (w + 10, 40), font, font_scale, text_color, font_thickness)
        cv2.putText(canvas, "Detected Corners", (2*w + 10, 40), font, font_scale, text_color, font_thickness)

        # Save
        output_path = output_dir / f"heatmap_{image_name}"
        cv2.imwrite(str(output_path), canvas)

    def test_single_image_detailed(self, inference, test_images, output_dir):
        """
        Generate detailed visualization for first test image

        Creates separate outputs:
        - Raw probability heatmap
        - Binary mask at different thresholds
        - Contour visualization
        """
        if not test_images:
            pytest.skip("No test images available")

        # Use first image
        image_path = test_images[0]
        image = cv2.imread(str(image_path))

        # Get predictions
        mask_prob, mask_resized = inference.predict_mask(image)

        # Save raw probability heatmap
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_dir / f"prob_heatmap_{image_path.name}"), heatmap)

        # Save masks at different thresholds
        thresholds = [0.3, 0.5, 0.7]
        for thresh in thresholds:
            binary_mask = (mask_resized > thresh).astype(np.uint8) * 255
            output_path = output_dir / f"mask_thresh{int(thresh*100)}_{image_path.name}"
            cv2.imwrite(str(output_path), binary_mask)

        print(f"\n  ✓ Detailed outputs for {image_path.name}")
        print(f"    - Probability heatmap")
        print(f"    - Binary masks at thresholds: {thresholds}")

    def test_model_confidence_stats(self, inference, test_images):
        """
        Compute and print confidence statistics across all test images
        """
        if not test_images:
            pytest.skip("No test images available")

        confidences = []
        detection_success = 0

        for image_path in test_images[:10]:  # Limit to first 10 for speed
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            mask_prob, _ = inference.predict_mask(image)
            corners = inference.detect_corners(image, debug=False)

            # Collect stats
            max_conf = mask_prob.max()
            mean_conf = mask_prob.mean()
            confidences.append((max_conf, mean_conf))

            if corners is not None:
                detection_success += 1

        # Print summary
        print("\n" + "="*60)
        print("Model Confidence Statistics")
        print("="*60)
        print(f"Images processed: {len(confidences)}")
        print(f"Detection success rate: {detection_success}/{len(confidences)} ({detection_success/len(confidences)*100:.1f}%)")
        print(f"\nConfidence ranges:")
        print(f"  Max confidence: {min(c[0] for c in confidences):.3f} - {max(c[0] for c in confidences):.3f}")
        print(f"  Mean confidence: {min(c[1] for c in confidences):.3f} - {max(c[1] for c in confidences):.3f}")
        print("="*60)
