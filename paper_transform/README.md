# Paper Transform Module

Geometry transformation module for paper documents. Provides perspective correction (warp) and automatic orientation detection.

## Features

- **Perspective Warp**: Straighten paper using detected corners
- **Auto-Orientation**: Automatically rotate document to correct reading direction
- **Clean API**: Simple functions that work with OpenCV images

## Installation

No additional dependencies required beyond the base project requirements (OpenCV, NumPy).

## Usage

### Basic Workflow

```python
import cv2
from paper_detection import PaperDetector
from paper_transform import warp_paper, auto_orient

# Load image
image = cv2.imread('photo.jpg')

# Detect paper corners
detector = PaperDetector()
corners = detector.detect(image)

if corners is not None:
    # Apply perspective correction
    warped = warp_paper(image, corners)

    # Auto-orient to correct reading direction
    oriented = auto_orient(warped)

    # Save result
    cv2.imwrite('result.jpg', oriented)
```

### Warp Only

```python
from paper_transform import warp_paper

# With auto-computed dimensions
warped = warp_paper(image, corners)

# With custom output size
warped = warp_paper(image, corners, dst_size=(1000, 1414))  # A4 aspect ratio
```

### Orientation Control

```python
from paper_transform import auto_orient

# Automatic orientation (default)
oriented = auto_orient(warped, orientation="auto")

# Force portrait (height > width)
oriented = auto_orient(warped, orientation="portrait")

# Force landscape (width > height)
oriented = auto_orient(warped, orientation="landscape")
```

### Manual Rotation

```python
from paper_transform.orient import rotate_180

# Rotate 180 degrees if document is upside down
flipped = rotate_180(oriented)
```

## API Reference

### `warp_paper(image_bgr, corners, dst_size=None)`

Apply perspective transformation to straighten paper document.

**Parameters:**
- `image_bgr` (np.ndarray): Input image in BGR format
- `corners` (np.ndarray): Array of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
- `dst_size` (tuple, optional): Output size (width, height). Auto-computed if None.

**Returns:**
- `np.ndarray`: Warped image in BGR format

**Raises:**
- `ValueError`: If image is None/empty or corners are invalid

### `auto_orient(image_bgr, orientation="auto")`

Automatically orient document to correct reading direction.

**Parameters:**
- `image_bgr` (np.ndarray): Input warped document in BGR format
- `orientation` (str): "auto", "portrait", or "landscape"

**Returns:**
- `np.ndarray`: Oriented image in BGR format

**Raises:**
- `ValueError`: If image is None/empty or orientation is invalid

### `rotate_180(image)`

Utility function to rotate image 180 degrees.

**Parameters:**
- `image` (np.ndarray): Input image

**Returns:**
- `np.ndarray`: Rotated image

## Testing

### Test with Ground Truth Data

```bash
python3 temp/test_transform.py
```

Tests warp and orientation using existing test images and ground truth corners.

### Test Full Pipeline

```bash
python3 temp/demo_full_pipeline.py images/photos/IMG_20230826_093159.jpg
```

Demonstrates the complete workflow: detection → warp → orientation.

## Architecture

### Module Structure

```
paper_transform/
├── __init__.py          # Public API exports
├── warp.py              # Perspective transformation
├── orient.py            # Orientation detection
└── README.md            # This file
```

### Design Principles

1. **Separation of Concerns**: Transform module is independent from detection
2. **No Detection**: Module only transforms, never detects corners
3. **Simple API**: Functions accept corners and images, return transformed images
4. **No OCR**: Text detection/OCR is out of scope for this module

### Corner Format

Corners can be in any order - they are automatically reordered internally to:
- top-left, top-right, bottom-right, bottom-left

This matches the output format from `PaperDetector.detect()`.

## Integration Example

```python
def process_document(image_path):
    """Complete document processing pipeline"""

    # Step 1: Load image
    image = cv2.imread(image_path)

    # Step 2: Detect paper
    detector = PaperDetector()
    corners = detector.detect(image)

    if corners is None:
        raise ValueError("Paper not detected")

    # Step 3: Warp
    warped = warp_paper(image, corners)

    # Step 4: Orient
    oriented = auto_orient(warped)

    return oriented
```

## Future Enhancements

Possible improvements for the auto-orientation function:

1. **Text-based orientation**: Use OCR to detect text direction
2. **ML-based orientation**: Train a model to recognize document orientation
3. **Confidence scores**: Return orientation confidence metrics
4. **Multi-page support**: Batch processing with consistent orientation

Currently, the module uses a simple aspect ratio heuristic which works well for most documents.

## Notes

- All functions work with BGR images (OpenCV standard format)
- White borders are added during warping to handle edge cases
- The module uses bilinear interpolation for high-quality results
- Auto-orientation prefers portrait mode (height > width)
