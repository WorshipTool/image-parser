"""
Tests for corner detection accuracy using ground truth data
"""

import cv2
import json
import numpy as np
import pytest
from pathlib import Path
from paper_detection import PaperDetector


class TestCornerAccuracy:
    """Test corner detection accuracy against ground truth"""

    @pytest.fixture
    def test_images_dir(self):
        """Path to test images directory"""
        return Path(__file__).parent / "test_images"

    @pytest.fixture
    def ground_truth(self):
        """Load ground truth corner data"""
        json_path = Path(__file__).parent / "test_corners_ground_truth.json"
        with open(json_path, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def reference_output_dir(self):
        """Directory for reference visualization images"""
        output_dir = Path(__file__).parent / "output" / "reference"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def visualize_ground_truth(
        self,
        image: np.ndarray,
        ground_truth_corners: np.ndarray,
        detected_corners: np.ndarray,
        output_path: Path
    ):
        """
        Create visualization showing ground truth vs detected corners.

        Ground truth: GREEN lines and circles
        Detected: BLUE lines and circles
        Difference: RED lines connecting matching corners
        """
        result = image.copy()
        h, w = result.shape[:2]

        # Scale parameters based on image size
        # Use shorter dimension as reference
        ref_size = min(h, w)

        # Scale thickness (3-20px depending on image size)
        line_thickness_gt = max(3, int(ref_size / 150))  # Green ground truth
        line_thickness_det = max(2, int(ref_size / 180))  # Blue detected
        line_thickness_err = max(2, int(ref_size / 200))  # Red error

        # Scale circle radius (8-30px)
        circle_radius_gt = max(8, int(ref_size / 100))
        circle_radius_det = max(6, int(ref_size / 120))

        # Scale font (0.5-3.0)
        font_scale = max(0.5, min(3.0, ref_size / 400))
        font_thickness = max(1, int(ref_size / 300))

        # Scale legend text
        legend_font_scale = max(0.6, min(2.5, ref_size / 500))
        legend_font_thickness = max(2, int(ref_size / 400))

        # Helper to clip line to image bounds
        def clip_line(p1, p2):
            """Simple clipping - just check if points are in bounds"""
            x1, y1 = p1
            x2, y2 = p2

            # If both points way outside, skip
            if (x1 < -1000 and x2 < -1000) or (x1 > w + 1000 and x2 > w + 1000):
                return None, None
            if (y1 < -1000 and y2 < -1000) or (y1 > h + 1000 and y2 > h + 1000):
                return None, None

            # Clip to reasonable bounds for drawing
            x1_clip = max(-10000, min(10000, x1))
            y1_clip = max(-10000, min(10000, y1))
            x2_clip = max(-10000, min(10000, x2))
            y2_clip = max(-10000, min(10000, y2))

            return (int(x1_clip), int(y1_clip)), (int(x2_clip), int(y2_clip))

        # Draw ground truth in GREEN
        for i in range(4):
            p1 = ground_truth_corners[i]
            p2 = ground_truth_corners[(i + 1) % 4]

            p1_clip, p2_clip = clip_line(p1, p2)
            if p1_clip and p2_clip:
                cv2.line(result, p1_clip, p2_clip, (0, 255, 0), line_thickness_gt)  # Green

        # Draw ground truth corners
        for i, corner in enumerate(ground_truth_corners):
            if 0 <= corner[0] < w and 0 <= corner[1] < h:
                pos = (int(corner[0]), int(corner[1]))
                cv2.circle(result, pos, circle_radius_gt, (0, 255, 0), -1)  # Green

                # Position label offset based on circle size
                label_offset = int(circle_radius_gt * 1.5)
                cv2.putText(
                    result,
                    f"GT{i}",
                    (pos[0] + label_offset, pos[1] + label_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    font_thickness
                )

        # Draw detected corners in BLUE
        for i in range(4):
            p1 = detected_corners[i]
            p2 = detected_corners[(i + 1) % 4]

            p1_clip, p2_clip = clip_line(p1, p2)
            if p1_clip and p2_clip:
                cv2.line(result, p1_clip, p2_clip, (255, 100, 0), line_thickness_det)  # Blue

        # Draw detected corners
        for i, corner in enumerate(detected_corners):
            if 0 <= corner[0] < w and 0 <= corner[1] < h:
                pos = (int(corner[0]), int(corner[1]))
                cv2.circle(result, pos, circle_radius_det, (255, 100, 0), -1)  # Blue

                # Position label to the left
                label_offset_x = int(circle_radius_det * 3)
                label_offset_y = int(circle_radius_det * 1.5)
                cv2.putText(
                    result,
                    f"D{i}",
                    (pos[0] - label_offset_x, pos[1] + label_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 100, 0),
                    font_thickness
                )

        # Draw RED lines showing difference between GT and detected
        for i in range(4):
            gt_corner = ground_truth_corners[i]
            det_corner = detected_corners[i]

            # Calculate distance
            distance = np.linalg.norm(gt_corner - det_corner)

            # Only draw if both corners are visible
            if (0 <= gt_corner[0] < w and 0 <= gt_corner[1] < h and
                0 <= det_corner[0] < w and 0 <= det_corner[1] < h):

                gt_pos = (int(gt_corner[0]), int(gt_corner[1]))
                det_pos = (int(det_corner[0]), int(det_corner[1]))

                # Draw red line showing error
                cv2.line(result, gt_pos, det_pos, (0, 0, 255), line_thickness_err)  # Red

                # Add distance label
                mid_x = int((gt_corner[0] + det_corner[0]) / 2)
                mid_y = int((gt_corner[1] + det_corner[1]) / 2)
                cv2.putText(
                    result,
                    f"{distance:.1f}px",
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.8,  # Slightly smaller for error labels
                    (0, 0, 255),
                    font_thickness
                )

        # Add legend (scaled spacing)
        legend_spacing = max(20, int(ref_size / 80))
        legend_y = max(20, int(ref_size / 100))
        legend_x = max(10, int(ref_size / 200))

        cv2.putText(result, "GREEN = Ground Truth (JSON)", (legend_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (0, 255, 0), legend_font_thickness)
        cv2.putText(result, "BLUE = Detected", (legend_x, legend_y + legend_spacing),
                   cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (255, 100, 0), legend_font_thickness)
        cv2.putText(result, "RED = Error", (legend_x, legend_y + legend_spacing * 2),
                   cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (0, 0, 255), legend_font_thickness)

        cv2.imwrite(str(output_path), result)

    @pytest.mark.parametrize("image_name", [
        "IMG_20230826_092914.jpg",
        "IMG_20230826_093159.jpg",
        "test_image_1.jpeg",
        "test_image_2.jpg",
    ])
    def test_corner_detection_accuracy(
        self,
        test_images_dir,
        ground_truth,
        reference_output_dir,
        image_name
    ):
        """Test that detected corners match ground truth within tolerance"""

        # Skip if no ground truth
        if image_name not in ground_truth:
            pytest.skip(f"No ground truth data for {image_name}")

        # Load ground truth
        gt_data = ground_truth[image_name]
        tolerance = gt_data["tolerance_pixels"]
        config = gt_data["config"]

        # Load image
        image_path = test_images_dir / image_name
        assert image_path.exists(), f"Image not found: {image_path}"

        image = cv2.imread(str(image_path))
        assert image is not None, f"Failed to load image: {image_path}"

        # Convert relative corners (0-1) to absolute pixels
        h, w = image.shape[:2]
        gt_corners_relative = np.array(gt_data["corners"], dtype=np.float32)
        gt_corners = gt_corners_relative.copy()
        gt_corners[:, 0] *= w  # x coordinates
        gt_corners[:, 1] *= h  # y coordinates

        # Detect corners
        detector = PaperDetector(**config)
        detected_corners = detector.detect(image)

        assert detected_corners is not None, f"Paper not detected in {image_name}"

        # Create reference visualization
        ref_output_path = reference_output_dir / f"ref_{image_name}"
        self.visualize_ground_truth(image, gt_corners, detected_corners, ref_output_path)

        # Compare each corner
        max_error = 0
        errors = []

        for i in range(4):
            gt_corner = gt_corners[i]
            det_corner = detected_corners[i]

            distance = np.linalg.norm(gt_corner - det_corner)
            errors.append(distance)
            max_error = max(max_error, distance)

            assert distance <= tolerance, (
                f"{image_name}: Corner {i} error = {distance:.2f}px "
                f"(tolerance = {tolerance}px)\n"
                f"  Ground truth: ({gt_corner[0]:.1f}, {gt_corner[1]:.1f})\n"
                f"  Detected:     ({det_corner[0]:.1f}, {det_corner[1]:.1f})\n"
                f"  Reference visualization: {ref_output_path}"
            )

        # Print summary
        avg_error = np.mean(errors)
        print(f"\n✓ {image_name}:")
        print(f"  Avg error: {avg_error:.2f}px")
        print(f"  Max error: {max_error:.2f}px")
        print(f"  Reference: {ref_output_path}")
