"""
Test suite for preprocess.py

Tests the preprocessing pipeline by processing sample images
and saving results to temp folder for manual verification.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

# Add parser to path
parser_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parser_dir))

from text_parser.preprocess import preprocess


@pytest.fixture(scope="module")
def test_images_dir():
    """Get test images directory"""
    return Path(__file__).parent / "images"


@pytest.fixture(scope="module")
def output_dir():
    """Create and return output directory for preprocessed images"""
    image_parser_dir = Path(__file__).parent.parent.parent.parent
    output_dir = image_parser_dir / "temp" / "preprocess_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(scope="module")
def test_images(test_images_dir):
    """Get list of test images"""
    images = sorted(test_images_dir.glob("*.jpg"))
    assert len(images) > 0, "No test images found"
    return images


def create_comparison_image(original: np.ndarray, preprocessed: np.ndarray) -> np.ndarray:
    """
    Create side-by-side comparison of original and preprocessed image

    Args:
        original: Original image (can be color or grayscale)
        preprocessed: Preprocessed image (grayscale/binary)

    Returns:
        Combined image showing original and preprocessed side by side
    """
    # Convert original to grayscale if needed for comparison
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original.copy()

    # Resize to match heights if needed
    h1, w1 = original_gray.shape
    h2, w2 = preprocessed.shape

    if h1 != h2:
        # Resize preprocessed to match original height
        scale = h1 / h2
        new_w2 = int(w2 * scale)
        preprocessed_resized = cv2.resize(preprocessed, (new_w2, h1), interpolation=cv2.INTER_CUBIC)
    else:
        preprocessed_resized = preprocessed
        new_w2 = w2

    # Create combined image
    combined = np.hstack([original_gray, preprocessed_resized])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 3
    color = 128  # Gray color for labels

    # Label for original
    cv2.putText(combined, "Original", (20, 60), font, font_scale, color, thickness)

    # Label for preprocessed
    cv2.putText(combined, "Preprocessed", (w1 + 20, 60), font, font_scale, color, thickness)

    # Add separator line
    cv2.line(combined, (w1, 0), (w1, h1), color, 2)

    return combined


def scale_image(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale image by given factor

    Args:
        img: Input image
        scale: Scale factor (e.g., 2.0 for 2x, 3.0 for 3x)

    Returns:
        Scaled image
    """
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def create_multi_scale_comparison(original: np.ndarray, scales: list[float]) -> np.ndarray:
    """
    Create side-by-side comparison of original and preprocessed images at multiple scales

    Args:
        original: Original image (can be color or grayscale)
        scales: List of scale factors to apply (e.g., [1.0, 3.0, 8.0])

    Returns:
        Combined image showing original and all preprocessed versions side by side
    """
    # Convert original to grayscale if needed
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original.copy()

    # Process image at each scale
    processed_images = []
    labels = ["Original"]

    # Add original
    processed_images.append(original_gray)

    # Process at each scale
    for scale in scales:
        # Scale image first
        scaled_img = scale_image(original, scale)
        # Then preprocess
        processed = preprocess(scaled_img, debug=False)
        processed_images.append(processed)
        labels.append(f"{scale}x scale")

    # Find target height (use original height)
    target_height = original_gray.shape[0]

    # Resize all images to same height
    resized_images = []
    x_positions = [0]  # Track x positions for separator lines
    current_x = 0

    for img in processed_images:
        if img.shape[0] != target_height:
            scale_factor = target_height / img.shape[0]
            new_width = int(img.shape[1] * scale_factor)
            img_resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
        else:
            img_resized = img

        resized_images.append(img_resized)
        current_x += img_resized.shape[1]
        x_positions.append(current_x)

    # Combine all images horizontally
    combined = np.hstack(resized_images)

    # Add labels and separators
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 3
    color = 128  # Gray color for labels

    for i, (label, x_pos) in enumerate(zip(labels, x_positions[:-1])):
        # Add label
        cv2.putText(combined, label, (x_pos + 20, 60), font, font_scale, color, thickness)

        # Add separator line (except before first image)
        if i > 0:
            cv2.line(combined, (x_pos, 0), (x_pos, target_height), color, 2)

    return combined


class TestPreprocess:
    """Test preprocessing functionality"""

    def test_preprocess_returns_valid_image(self, test_images):
        """Test that preprocess returns a valid processed image"""
        img_path = test_images[0]
        original = cv2.imread(str(img_path))

        preprocessed = preprocess(original, debug=False)

        # Check that result is valid grayscale image
        assert preprocessed is not None, "Preprocess returned None"
        assert len(preprocessed.shape) == 2, "Output should be 2D grayscale"
        assert preprocessed.dtype == np.uint8, "Output should be uint8"

    def test_preprocess_handles_grayscale_input(self, test_images):
        """Test that preprocess handles grayscale input"""
        img_path = test_images[0]
        original = cv2.imread(str(img_path))
        gray_input = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        preprocessed = preprocess(gray_input, debug=False)

        # Should still work and return valid image
        assert preprocessed is not None
        assert len(preprocessed.shape) == 2

    def test_preprocess_all_images_and_save(self, test_images, output_dir):
        """
        Process all test images and save preprocessed results for manual verification

        This test processes each image and saves:
        - Individual preprocessed images
        - Side-by-side comparison images
        """
        print("\n" + "="*80)
        print("PREPROCESSING TEST - All Images")
        print("="*80)
        print(f"\nOutput directory: {output_dir}")

        for img_path in test_images:
            print(f"\n{'─'*80}")
            print(f"Processing: {img_path.name}")

            # Load original
            original = cv2.imread(str(img_path))
            assert original is not None, f"Failed to load {img_path}"
            print(f"  ✓ Loaded: {original.shape}")

            # Preprocess
            preprocessed = preprocess(original, debug=False)
            assert preprocessed is not None, "Preprocessing failed"
            print(f"  ✓ Preprocessed: {preprocessed.shape}")

            # Verify output statistics
            unique_vals = len(np.unique(preprocessed))

            # Save preprocessed
            preprocessed_path = output_dir / f"preprocessed_{img_path.name}"
            cv2.imwrite(str(preprocessed_path), preprocessed)
            print(f"  ✓ Saved: {preprocessed_path.name}")

            # Create and save comparison
            comparison = create_comparison_image(original, preprocessed)
            comparison_path = output_dir / f"comparison_{img_path.name}"
            cv2.imwrite(str(comparison_path), comparison)
            print(f"  ✓ Saved: {comparison_path.name}")

            # Print stats
            scale = preprocessed.shape[0] / original.shape[0]
            print(f"  ✓ Scale: {scale:.1f}x, Unique values: {unique_vals}")

        print("\n" + "="*80)
        print(f"All preprocessed images saved to: {output_dir}")
        print("="*80)

    def test_preprocess_debug_mode(self, test_images, output_dir):
        """Test that debug mode saves images correctly"""
        img_path = test_images[0]
        original = cv2.imread(str(img_path))

        # Run with debug mode
        preprocessed = preprocess(original, debug=True)

        # Should still return valid result
        assert preprocessed is not None
        assert len(preprocessed.shape) == 2

    def test_preprocess_multi_scale_comparison(self, test_images, output_dir):
        """
        Test preprocessing with multiple scale factors and save comparison

        This test processes each image with scales 1x, 3x, 8x and creates
        a side-by-side comparison showing: Original | 1x | 3x | 8x
        """
        scales = [1.0, 3.0, 5.0]

        print("\n" + "="*80)
        print("MULTI-SCALE PREPROCESSING TEST")
        print("="*80)
        print(f"\nScale factors: {scales}")
        print(f"Output directory: {output_dir}")

        for img_path in test_images:
            print(f"\n{'─'*80}")
            print(f"Processing: {img_path.name}")

            # Load original
            original = cv2.imread(str(img_path))
            assert original is not None, f"Failed to load {img_path}"
            print(f"  ✓ Loaded: {original.shape}")

            # Create multi-scale comparison
            comparison = create_multi_scale_comparison(original, scales)
            print(f"  ✓ Created comparison: {comparison.shape}")

            # Save multi-scale comparison
            comparison_path = output_dir / f"multiscale_{img_path.name}"
            cv2.imwrite(str(comparison_path), comparison)
            print(f"  ✓ Saved: {comparison_path.name}")

        print("\n" + "="*80)
        print(f"All multi-scale comparisons saved to: {output_dir}")
        print("="*80)
