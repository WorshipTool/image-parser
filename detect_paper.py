#!/usr/bin/env python3
"""
Simple script to detect paper in an image and display metrics
Usage: python3 detect_paper.py <path_to_image>
"""

import sys
import cv2
from pathlib import Path
from paper_detection import PaperDetector, PaperVisualizer


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect_paper.py <path_to_image>")
        print("Example: python3 detect_paper.py images/photos/IMG_20230826_093159.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        sys.exit(1)

    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]} px")

    # Try different detection strategies
    strategies = [
        {'brightness_threshold': 200, 'min_area_ratio': 0.01},
        {'brightness_threshold': 180, 'min_area_ratio': 0.005},
        {'brightness_threshold': 120, 'min_area_ratio': 0.003},
        {'brightness_threshold': 100, 'min_area_ratio': 0.003},
    ]

    corners = None
    detector = None

    print("Detecting paper...")
    for i, params in enumerate(strategies, 1):
        detector = PaperDetector(**params)
        corners = detector.detect(image)
        if corners is not None:
            print(f"✓ Paper detected (strategy {i})")
            break

    if corners is None:
        print("✗ Paper was not detected!")
        sys.exit(1)

    # Calculate metrics
    metrics = detector.get_paper_metrics(corners, image.shape)

    # Print metrics
    print("\nPaper Metrics:")
    print(f"  Cover Ratio:      {metrics['cover_ratio']*100:.1f}%")
    print(f"  Rectangularity:   {metrics['rectangularity']*100:.1f}%")
    print(f"  Angle:            {metrics['angle']:.1f}°")
    print(f"  Perspective:      {metrics['perspective_angle']:.1f}°")

    # Visualize
    visualizer = PaperVisualizer()
    result = visualizer.visualize_with_metrics(image, corners, metrics)

    # Save result
    output_path = image_path.parent / f"detected_{image_path.name}"
    cv2.imwrite(str(output_path), result)
    print(f"\n✓ Result saved: {output_path}")


if __name__ == "__main__":
    main()
