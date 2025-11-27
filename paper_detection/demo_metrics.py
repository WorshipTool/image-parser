"""
Demo script to showcase paper metrics calculation
"""

import cv2
from pathlib import Path
from paper_detection import PaperDetector, PaperVisualizer


def test_metrics_on_image(image_path: Path, detector: PaperDetector, visualizer: PaperVisualizer):
    """Test metrics calculation on a single image"""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path.name}")
    print(f"{'='*60}")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]} px")

    # Detect paper
    corners = detector.detect(image)

    if corners is None:
        print("Paper was not detected!")
        return

    print("Paper successfully detected!")

    # Calculate metrics
    metrics = detector.get_paper_metrics(corners, image.shape)

    # Print metrics
    print("\nPaper Metrics:")
    print(f"  Cover Ratio:      {metrics['cover_ratio']*100:.1f}% (how much of image is paper)")
    print(f"  Rectangularity:   {metrics['rectangularity']*100:.1f}% (how rectangular is shape)")
    print(f"  Angle:            {metrics['angle']:.1f}° (rotation from horizontal)")
    print(f"  Perspective:      {metrics['perspective_angle']:.1f}° (perspective distortion)")

    # Interpret metrics
    print("\nInterpretation:")

    # Cover ratio
    if metrics['cover_ratio'] > 0.7:
        print("  → Large paper, fills most of image")
    elif metrics['cover_ratio'] > 0.3:
        print("  → Medium-sized paper")
    else:
        print("  → Small paper in frame")

    # Rectangularity
    if metrics['rectangularity'] > 0.95:
        print("  → Very rectangular shape (likely scan or flat)")
    elif metrics['rectangularity'] > 0.85:
        print("  → Good rectangular shape")
    else:
        print("  → Shape deviates from rectangle")

    # Angle
    if abs(metrics['angle']) < 5:
        print("  → Nearly horizontal/vertical alignment")
    elif abs(metrics['angle']) < 45:
        print("  → Moderate rotation")
    else:
        print("  → Significant rotation")

    # Perspective
    if metrics['perspective_angle'] < 3:
        print("  → Minimal perspective (likely scan or overhead shot)")
    elif metrics['perspective_angle'] < 10:
        print("  → Some perspective distortion")
    else:
        print("  → Significant perspective (angled photo)")

    # Visualize with metrics
    result = visualizer.visualize_with_metrics(image, corners, metrics)

    # Save result
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"metrics_{image_path.name}"
    cv2.imwrite(str(output_path), result)
    print(f"\nVisualization saved: {output_path}")


def main():
    """Main demo function"""

    # Create detector and visualizer
    detector = PaperDetector(brightness_threshold=180, min_area_ratio=0.005)
    visualizer = PaperVisualizer()

    # Test on original test image
    original_image = Path(__file__).parent.parent / "images" / "photos" / "IMG_20230826_093159.jpg"

    if original_image.exists():
        test_metrics_on_image(original_image, detector, visualizer)

    # Test on new test images
    test_images_dir = Path(__file__).parent / "tests" / "test_images"

    if test_images_dir.exists():
        test_images = sorted(test_images_dir.glob("*.jpg")) + sorted(test_images_dir.glob("*.jpeg"))

        for image_path in test_images:
            # Use adaptive detection for challenging images
            strategies = [
                {'brightness_threshold': 180, 'min_area_ratio': 0.005},
                {'brightness_threshold': 120, 'min_area_ratio': 0.003},
                {'brightness_threshold': 100, 'min_area_ratio': 0.003},
            ]

            for params in strategies:
                detector = PaperDetector(**params)
                image = cv2.imread(str(image_path))
                if image is not None and detector.detect(image) is not None:
                    break

            test_metrics_on_image(image_path, detector, visualizer)

    print(f"\n{'='*60}")
    print("Demo completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
