# Document Orientation Submodule

Text-based automatic orientation detection using OCR. Part of the `paper_detection` module.

## Quick Start

```python
from paper_detection import PaperDetector
from paper_detection.document_orient import orient_by_text
from paper_transform import warp_paper

# Detect and warp
detector = PaperDetector()
corners = detector.detect(image)
warped = warp_paper(image, corners)

# Orient based on text
oriented = orient_by_text(warped)
```

## How It Works

1. Tests all 4 rotations (0°, 90°, 180°, 270°)
2. Runs OCR on each rotation
3. Scores text quality:
   - `score = text_length * purity`
   - `purity = good_chars / total_chars`
4. Returns rotation with highest score
5. Falls back to geometric orientation if OCR fails

## Requirements

### System Package
```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr
```

### Python Package
Already in `requirements.txt`:
- `pytesseract==0.3.10`

## API

### Main Function

```python
orient_by_text(
    image_bgr: np.ndarray,
    ocr_engine: str = "tesseract",
    ocr_kwargs: Optional[dict] = None,
    min_score_threshold: float = 5.0,
    debug: bool = False
) -> np.ndarray
```

### Batch Processing

```python
batch_orient_text(
    images: list,
    ocr_engine: str = "tesseract",
    ocr_kwargs: Optional[dict] = None,
    debug: bool = False
) -> list
```

## Examples

### With Debug Info

```python
oriented = orient_by_text(warped, debug=True)
```

Output:
```
============================================================
Text Orientation Detection - Debug Info
============================================================

Rotation   0°: score =   45.32
  Text preview: This is some example text...

Rotation  90°: score =    3.21
  Text preview: ▄▀▄▀ ▄▀▄▀ ▄▀▄▀...

Best rotation: 0° (score: 45.32)
============================================================
```

### Different Languages

```python
# Czech
oriented = orient_by_text(warped, ocr_kwargs={'lang': 'ces'})

# Multiple languages
oriented = orient_by_text(warped, ocr_kwargs={'lang': 'eng+ces'})
```

### Using EasyOCR

```python
# Requires: pip install easyocr
oriented = orient_by_text(warped, ocr_engine="easyocr")
```

## Performance

- **Speed**: 2-5 seconds per image (4 OCR runs)
- **Accuracy**: Excellent for text documents
- **Fallback**: Fast geometric orientation if OCR fails

## Testing

```bash
# Run unit tests
python3 -m pytest paper_detection/document_orient/tests/ -v

# Test import
python3 -c "from paper_detection.document_orient import orient_by_text"
```

## Module Structure

```
document_orient/
├── __init__.py           # Public API
├── orient_text.py        # Main orientation logic
├── ocr_engines.py        # OCR abstraction
└── tests/
    └── test_orient_text.py
```

## Comparison with Geometric Orientation

| Feature | Geometric | Text-based |
|---------|-----------|------------|
| Speed | Instant | 2-5 sec |
| Accuracy (text docs) | Good | Excellent |
| Dependencies | None | Tesseract |

## See Also

- `../../paper_transform/orient.py` - Geometric orientation (aspect ratio)
- `../../paper_transform/warp.py` - Perspective correction
- `../detector.py` - Paper corner detection

## Demo Scripts

```bash
# Full pipeline demo
python3 temp/demo_text_orientation.py images/photos/IMG_20230826_093159.jpg

# Compare geometric vs text-based
python3 temp/example_integration.py images/photos/IMG_20230826_093159.jpg
```
