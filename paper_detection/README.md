# Paper Detection Module

Isolated module for paper detection in images using OpenCV.

## Features

- Paper detection in photos using edge detection and contours
- Visualization with blue frame around detected paper
- Light blue transparent overlay on paper
- Calculation of detected paper dimensions
- Complete test suite

## Usage

### Basic Detection

```python
import cv2
from paper_detection import PaperDetector

# Load image
image = cv2.imread("path/to/image.jpg")

# Create detector
detector = PaperDetector()

# Detect paper
corners = detector.detect(image)

if corners is not None:
    print("Paper detected!")
    print(f"Corners: {corners}")

    # Calculate dimensions
    width, height = detector.get_paper_dimensions(corners)
    print(f"Dimensions: {width}x{height} px")
else:
    print("Paper not found")
```

### Visualization

```python
import cv2
from paper_detection import PaperDetector, PaperVisualizer

# Load and detect
image = cv2.imread("path/to/image.jpg")
detector = PaperDetector()
corners = detector.detect(image)

if corners is not None:
    # Create visualizer
    visualizer = PaperVisualizer()

    # Basic visualization
    result = visualizer.visualize(image, corners)
    cv2.imwrite("output.jpg", result)

    # Visualization with info
    result_with_info = visualizer.visualize_with_info(image, corners)
    cv2.imwrite("output_with_info.jpg", result_with_info)

    # Side-by-side comparison
    comparison = visualizer.create_side_by_side(image, result)
    cv2.imwrite("comparison.jpg", comparison)
```

### Custom Parameters

```python
# Detector with custom parameters
detector = PaperDetector(
    brightness_threshold=180,  # Lower threshold for darker paper
    min_area_ratio=0.05,      # Larger minimum area
    approx_epsilon=0.02,
    use_threshold_method=True # Use threshold method (default)
)

# Visualizer with custom colors
visualizer = PaperVisualizer(
    border_color=(0, 255, 0),      # Green frame (BGR)
    border_thickness=5,
    overlay_color=(0, 200, 255),   # Yellow overlay (BGR)
    overlay_alpha=0.5              # More transparent
)
```

## Running Tests

```bash
# From module root directory
pytest paper_detection/tests/test_detector.py -v

# With coverage
pytest paper_detection/tests/test_detector.py --cov=paper_detection -v

# Specific test
pytest paper_detection/tests/test_detector.py::TestPaperDetector::test_detect_paper -v
```

## Module Structure

```
paper_detection/
├── __init__.py          # Module exports
├── detector.py          # PaperDetector class
├── visualizer.py        # PaperVisualizer class
├── README.md           # This documentation
└── tests/
    ├── __init__.py
    ├── test_detector.py # Tests
    └── output/         # Output images from tests
```

## Detector Parameters

- `brightness_threshold`: Brightness threshold for white paper detection, 0-255 (default: 200)
- `min_area_ratio`: Minimum ratio of paper area to total image (default: 0.01)
- `approx_epsilon`: Epsilon for polygon approximation as ratio of perimeter (default: 0.02)
- `use_threshold_method`: Use threshold method instead of Canny edge detection (default: True)

## Visualizer Parameters

- `border_color`: Frame color in BGR format (default: (255, 100, 0) - blue)
- `border_thickness`: Frame thickness in pixels (default: 3)
- `overlay_color`: Transparent overlay color in BGR (default: (255, 200, 100) - light blue)
- `overlay_alpha`: Overlay transparency 0.0-1.0 (default: 0.3)

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- pytest (for tests)
