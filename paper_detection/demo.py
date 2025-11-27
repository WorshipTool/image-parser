"""
Demo script to showcase paper_detection module usage
"""

import cv2
from pathlib import Path
from paper_detection import PaperDetector, PaperVisualizer


def main():
    """Main demo function"""

    # Path to test image
    image_path = Path(__file__).parent.parent / "images" / "photos" / "IMG_20230826_093159.jpg"

    if not image_path.exists():
        print(f"Error: Image not found at path: {image_path}")
        return

    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))

    if image is None:
        print("Error: Failed to load image")
        return

    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]} px")

    # Create detector
    print("\nCreating detector...")
    detector = PaperDetector()

    # Detect paper
    print("Detecting paper...")
    corners = detector.detect(image)

    if corners is None:
        print("Paper was not detected!")
        return

    print("Paper successfully detected!")
    print(f"Paper corners:")
    for i, corner in enumerate(corners):
        corner_names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
        print(f"  {corner_names[i]}: ({corner[0]:.1f}, {corner[1]:.1f})")

    # Calculate dimensions
    width, height = detector.get_paper_dimensions(corners)
    print(f"\nPaper dimensions: {width}x{height} px")

    # Calculate area
    import numpy as np
    area = cv2.contourArea(corners.astype(np.int32))
    print(f"Paper area: {int(area)} px²")

    # Create visualizer
    print("\nCreating visualization...")
    visualizer = PaperVisualizer()

    # Basic visualization
    result = visualizer.visualize(image, corners)

    # Visualization with info
    result_with_info = visualizer.visualize_with_info(image, corners)

    # Side-by-side comparison
    comparison = visualizer.create_side_by_side(image, result_with_info)

    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Save results
    output_basic = output_dir / "demo_basic.jpg"
    output_info = output_dir / "demo_with_info.jpg"
    output_comparison = output_dir / "demo_comparison.jpg"

    cv2.imwrite(str(output_basic), result)
    cv2.imwrite(str(output_info), result_with_info)
    cv2.imwrite(str(output_comparison), comparison)

    print(f"\nResults saved:")
    print(f"  {output_basic}")
    print(f"  {output_info}")
    print(f"  {output_comparison}")

    # Display result (if display is available)
    try:
        print("\nDisplaying result (press any key to exit)...")
        cv2.imshow("Comparison - Original vs Detected", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("\nCannot display window (running in headless mode)")

    print("\nDemo completed!")


if __name__ == "__main__":
    main()
