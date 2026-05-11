# Parser Module

Intelligent sheet music extraction from images.

## Overview

Automatically extracts sheet music using multi-stage detection:
- **Photos with paper** → Paper detection + perspective correction + orientation
- **Screenshots** → YOLO sheet detection + merged cropping (all sheets combined)
- **No detection** → Returns original image

## Quick Start

```python
from parser.parse import get_sheet_components_from_image

# Extract sheets from image
sheets = get_sheet_components_from_image("image.jpg", debug=True)

# Save result
import cv2
cv2.imwrite("output.jpg", sheets[0])
```

## Pipeline

1. **Paper Detection** - U-Net checks for physical paper (shouldCrop decision)
2. **Perspective Correction** - Warps paper to rectangular view (if paper detected)
3. **Orientation** - OCR-based rotation detection (if paper detected)
4. **Sheet Detection** - YOLO detects sheets, merges bounding boxes, crops (if no paper detected)

## API

**`get_sheet_components_from_image(image, debug=False)`**
- `image`: Path (str/Path) or numpy array (BGR)
- Returns: `list[np.ndarray]` - Always returns list with 1 sheet image
  - Paper detected: transformed sheet
  - Screenshot: merged bounding box of all detected sheets
  - No detection: original image

**`get_sheet_components_batch(image_paths, debug=False, output_dir=None)`**
- `image_paths`: List of image paths
- `output_dir`: Optional output directory
- Returns: `list[np.ndarray]` - All extracted sheets

## Setup

```bash
# Install dependencies
pip install torch torchvision opencv-python pytesseract ultralytics

# Download YOLO model
python parser/sheet_detection/prepare.py
```

Models auto-load on import if they exist.

## CLI Usage

```bash
# Single image
python parser/parse.py image.jpg

# Multiple images
python parser/parse.py img1.jpg img2.jpg img3.jpg

# With output directory
python parser/parse.py *.jpg -o output/

# With debug info
python parser/parse.py image.jpg -o output/ --debug
```

## Examples

**Photo with paper:**
```python
sheets = get_sheet_components_from_image("photo.jpg", debug=True)
# → Paper detected → Warped → Oriented → 1 sheet
```

**Screenshot:**
```python
sheets = get_sheet_components_from_image("screenshot.png", debug=True)
# → No paper → YOLO detection → 3 sheets merged into 1 cropped image
```

**Batch:**
```python
from parser.parse import get_sheet_components_batch
from pathlib import Path

sheets = get_sheet_components_batch(
    list(Path("input").glob("*.jpg")),
    output_dir=Path("output")
)
```

## Structure

```
parser/
├── parse.py              # Main API
├── paper_detection/      # U-Net paper detection
├── paper_transform/      # Perspective correction
└── sheet_detection/      # YOLO sheet detection
```

## Testing

```bash
pytest parser/paper_detection/tests/ -v
pytest parser/paper_transform/tests/ -v
pytest parser/sheet_detection/tests/ -v
```

## Submodules

- **`paper_detection/`** - U-Net segmentation for physical paper
- **`paper_transform/`** - Perspective correction and orientation
- **`sheet_detection/`** - YOLOv8 object detection (see [`sheet_detection/README.md`](sheet_detection/README.md))

## Performance

~0.5-1.5s per image (CPU)
