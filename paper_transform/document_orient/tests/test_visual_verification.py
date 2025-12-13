"""
Visual verification tests for orientation

Generates side-by-side visualizations to manually verify orientation quality.
"""

import pytest
import cv2
import numpy as np
import json
from pathlib import Path

from paper_transform.document_orient import orient_by_text


class TestVisualVerification:
    """Generate visual verification outputs for manual inspection"""

    @pytest.fixture
    def test_data_dir(self):
        """Get test data directory"""
        return Path(__file__).parent / "test_data"

    @pytest.fixture
    def output_dir(self):
        """Output directory for visualizations"""
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    @pytest.fixture
    def ground_truth(self, test_data_dir):
        """Load ground truth annotations"""
        ground_truth_file = test_data_dir / "ground_truth.json"

        if not ground_truth_file.exists():
            pytest.skip("Ground truth file not found")

        with open(ground_truth_file, 'r') as f:
            return json.load(f)

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

    def _create_visualization(
        self,
        original: np.ndarray,
        misoriented: np.ndarray,
        oriented: np.ndarray,
        image_name: str,
        expected_rotation: int,
        description: str,
        output_path: Path
    ):
        """
        Create 3-column visualization showing orientation process

        Layout: Original (correct) | Misoriented | Auto-oriented
        """
        # Scale all to same height
        target_height = 800
        images_to_display = []
        labels = []

        for img, label in [
            (original, f"Ground Truth\n(correct orientation)"),
            (misoriented, f"Misoriented\n(rotated {(360 - expected_rotation) % 360}° from correct)"),
            (oriented, f"Auto-Oriented\n(result from orient_by_text)")
        ]:
            h, w = img.shape[:2]
            scale = target_height / h
            new_w = int(w * scale)
            scaled = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
            images_to_display.append(scaled)
            labels.append(label)

        # Calculate total width with gaps
        widths = [img.shape[1] for img in images_to_display]
        total_width = sum(widths) + 40  # 20px gaps
        header_height = 120

        # Create canvas with header space
        canvas = np.ones((target_height + header_height, total_width, 3), dtype=np.uint8) * 255

        # Add title
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = f"Orientation Test: {image_name}"
        cv2.putText(canvas, title, (20, 35), font, 1.0, (0, 0, 0), 2)

        subtitle = f"Expected rotation: {expected_rotation}° | {description}"
        cv2.putText(canvas, subtitle, (20, 70), font, 0.6, (100, 100, 100), 1)

        # Place images with gaps
        x_offset = 0
        for i, (img, label) in enumerate(zip(images_to_display, labels)):
            w = img.shape[1]

            # Place image
            canvas[header_height:header_height + target_height, x_offset:x_offset + w] = img

            # Add label above image
            # Split label by newlines
            label_lines = label.split('\n')
            y_pos = header_height - 60
            for line in label_lines:
                cv2.putText(canvas, line, (x_offset + 10, y_pos), font, 0.5, (0, 0, 0), 1)
                y_pos += 20

            x_offset += w + 20  # 20px gap

        # Save
        cv2.imwrite(str(output_path), canvas)

    def test_generate_orientation_visualizations(self, test_data_dir, ground_truth, output_dir):
        """
        Generate visual comparisons for all test images

        Creates side-by-side visualizations showing:
        1. Original (correct orientation)
        2. Misoriented version (as it would appear if scanned incorrectly)
        3. Auto-oriented result (what orient_by_text produces)

        This allows manual verification of orientation quality.
        """
        print("\n" + "=" * 70)
        print("Generating Orientation Visualizations")
        print("=" * 70)

        generated = 0

        for image_name, annotations in ground_truth.items():
            image_path = test_data_dir / image_name

            if not image_path.exists():
                print(f"⚠ Skipping {image_name} - file not found")
                continue

            # Load image (this is correctly oriented)
            original = cv2.imread(str(image_path))
            expected_rotation = annotations["expected_rotation"]
            description = annotations["description"]

            # Create misoriented version
            # If expected rotation is 180, we rotate by 180 to simulate upside down
            inverse_rotation = (360 - expected_rotation) % 360
            misoriented = self._apply_rotation(original, inverse_rotation)

            # Auto-orient
            try:
                oriented = orient_by_text(misoriented, debug=False)

                # Create visualization
                output_name = f"orientation_test_{image_name}"
                output_path = output_dir / output_name

                self._create_visualization(
                    original,
                    misoriented,
                    oriented,
                    image_name,
                    expected_rotation,
                    description,
                    output_path
                )

                print(f"✓ {output_name}")
                generated += 1

            except Exception as e:
                print(f"✗ {image_name} - Error: {e}")

        print("\n" + "=" * 70)
        print(f"✓ Generated {generated} visualizations in {output_dir}")
        print("=" * 70)

        assert generated > 0, "No visualizations generated"

    def test_generate_rotation_matrix(self, test_data_dir, ground_truth, output_dir):
        """
        Generate a matrix showing each test image in all 4 rotations

        This helps verify the ground truth annotations are correct.
        For each test image, shows: 0°, 90°, 180°, 270° rotations.
        """
        print("\n" + "=" * 70)
        print("Generating Rotation Matrix")
        print("=" * 70)

        for image_name, annotations in ground_truth.items():
            image_path = test_data_dir / image_name

            if not image_path.exists():
                continue

            # Load image
            image = cv2.imread(str(image_path))
            expected_rotation = annotations["expected_rotation"]

            # Create all 4 rotations
            rotations = {
                0: image,
                90: self._apply_rotation(image, 90),
                180: self._apply_rotation(image, 180),
                270: self._apply_rotation(image, 270)
            }

            # Scale all to fixed height
            target_height = 400
            scaled_images = {}

            for angle, img in rotations.items():
                h, w = img.shape[:2]
                scale = target_height / h
                new_w = int(w * scale)
                scaled = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
                scaled_images[angle] = scaled

            # Create 2x2 grid
            row1_width = scaled_images[0].shape[1] + scaled_images[90].shape[1] + 20
            row2_width = scaled_images[180].shape[1] + scaled_images[270].shape[1] + 20
            max_width = max(row1_width, row2_width) + 40  # Extra padding

            header_height = 80
            gap = 20
            canvas_height = header_height + target_height * 2 + gap * 3
            canvas = np.ones((canvas_height, max_width, 3), dtype=np.uint8) * 255

            # Add title
            font = cv2.FONT_HERSHEY_SIMPLEX
            title = f"{image_name} - All Rotations (Expected: {expected_rotation}°)"
            cv2.putText(canvas, title, (20, 40), font, 0.7, (0, 0, 0), 2)

            # Place images in grid
            y_offset = header_height + gap

            # Row 1: 0° and 90°
            x_offset = gap
            for angle in [0, 90]:
                img = scaled_images[angle]
                h_img, w_img = img.shape[:2]

                # Ensure canvas is wide enough
                if x_offset + w_img > canvas.shape[1]:
                    # Expand canvas width
                    new_canvas = np.ones((canvas.shape[0], x_offset + w_img + gap, 3), dtype=np.uint8) * 255
                    new_canvas[:, :canvas.shape[1]] = canvas
                    canvas = new_canvas

                # Highlight expected rotation
                color = (0, 255, 0) if angle == expected_rotation else (200, 200, 200)
                border_thickness = 4 if angle == expected_rotation else 2

                # Draw border
                cv2.rectangle(
                    canvas,
                    (x_offset - 2, y_offset - 2),
                    (x_offset + w_img + 2, y_offset + h_img + 2),
                    color,
                    border_thickness
                )

                # Place image
                canvas[y_offset:y_offset + h_img, x_offset:x_offset + w_img] = img

                # Label
                label = f"{angle}°" + (" ✓" if angle == expected_rotation else "")
                cv2.putText(canvas, label, (x_offset + 5, y_offset + 30), font, 0.8, color, 2)

                x_offset += w_img + gap

            # Row 2: 180° and 270°
            y_offset += target_height + gap
            x_offset = gap
            for angle in [180, 270]:
                img = scaled_images[angle]
                h_img, w_img = img.shape[:2]

                # Ensure canvas is wide enough
                if x_offset + w_img > canvas.shape[1]:
                    # Expand canvas width
                    new_canvas = np.ones((canvas.shape[0], x_offset + w_img + gap, 3), dtype=np.uint8) * 255
                    new_canvas[:, :canvas.shape[1]] = canvas
                    canvas = new_canvas

                # Highlight expected rotation
                color = (0, 255, 0) if angle == expected_rotation else (200, 200, 200)
                border_thickness = 4 if angle == expected_rotation else 2

                # Draw border
                cv2.rectangle(
                    canvas,
                    (x_offset - 2, y_offset - 2),
                    (x_offset + w_img + 2, y_offset + h_img + 2),
                    color,
                    border_thickness
                )

                # Place image
                canvas[y_offset:y_offset + h_img, x_offset:x_offset + w_img] = img

                # Label
                label = f"{angle}°" + (" ✓" if angle == expected_rotation else "")
                cv2.putText(canvas, label, (x_offset + 5, y_offset + 30), font, 0.8, color, 2)

                x_offset += w_img + gap

            # Save
            output_name = f"rotation_matrix_{image_name}"
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), canvas)

            print(f"✓ {output_name}")

        print("\n" + "=" * 70)
        print(f"✓ Rotation matrices saved to {output_dir}")
        print("=" * 70)
