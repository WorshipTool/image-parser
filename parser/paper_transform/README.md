# Paper Transform

Straighten and orient paper documents using perspective correction and OCR.

## Quick Start

```python
from paper_transform import warp_paper, orient_by_text

# Straighten paper
warped = warp_paper(image, corners)

# Auto-rotate using OCR
oriented_img, angle, confidence = orient_by_text(warped)
```

## API

### `warp_paper(image_bgr, corners, dst_size=None)`
Perspective correction. Returns straightened image.

### `orient_by_text(image)`
OCR-based orientation. Returns `(image, angle, confidence)` or `None`.
Tests all 4 rotations, picks best. ~2-5 seconds.

## Example

```python
import cv2
from paper_detection import PaperDetector
from paper_transform import warp_paper, orient_by_text

image = cv2.imread('photo.jpg')
corners = PaperDetector().detect(image)

warped = warp_paper(image, corners)
result = orient_by_text(warped)

if result:
    oriented_img, angle, conf = result
    cv2.imwrite('output.jpg', oriented_img)
```

## Testing

```bash
pytest paper_transform/tests/ -v
```

## Dependencies

- OpenCV, NumPy, pytesseract, Tesseract OCR

```bash
brew install tesseract  # macOS
```
