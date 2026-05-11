# Text Parser Module

OCR-based text extraction and chord parsing for sheet music images.

## Features

- **OCR Reading**: Extracts text from images using Pytesseract (Czech/Slovak support)
- **Chord Detection**: Identifies and formats chord notation (e.g., `[Am]`, `[G7]`)
- **Section Grouping**: Automatically detects and labels sections (V1, V2, Chorus, etc.)
- **Title Extraction**: Identifies song title from image content
- **No Preprocessing**: Reads images as-is without paper detection or transformation

## Usage

### Command Line

```bash
# Basic usage - print to console
python -m parser.text_parser sheet.jpg

# With debug output
python -m parser.text_parser sheet.jpg --debug

# Save to JSON file
python -m parser.text_parser sheet.jpg -o output.json

# Save to formatted text file
python -m parser.text_parser sheet.jpg -o output.txt --format txt
```

### Python API

```python
from parser.text_parser import read_and_parse_image

# Parse image from file path
result = read_and_parse_image("sheet.jpg", debug=True)

# Parse image from numpy array
import cv2
image_bgr = cv2.imread("sheet.jpg")
result = read_and_parse_image(image_bgr, debug=False)

# Access results
print(result['title'])          # Song title
print(result['data'])           # Formatted text with chords
print(result['inputImagePath']) # Source image path
```

## Output Format

### JSON Output

```json
{
  "title": "Amazing Grace",
  "data": "{V1}[Am]Amazing grace...\\n{Chorus1}[G]How sweet...",
  "inputImagePath": "/path/to/sheet.jpg"
}
```

### Text Output

```
Title: Amazing Grace
================================================================================

{V1}[Am]Amazing grace, how [G]sweet the sound
That [C]saved a wretch like [Am]me

{Chorus1}[G]How sweet the [C]sound
```

## Chord Format

Chords are inserted inline with lyrics based on horizontal position:
- `[Am]` - A minor chord
- `[G7]` - G7 chord
- `[D/F#]` - D over F# bass note

## Section Detection

Sections are automatically detected by line spacing and labeled:
- `{V1}`, `{V2}`, etc. - Verses
- `{Chorus1}`, `{Chorus2}`, etc. - Choruses
- Sections numbered sequentially

## Requirements

- Python 3.7+
- OpenCV (cv2)
- Pytesseract with Czech/Slovak language data
- NumPy

## Module Structure

```
text_parser/
├── __init__.py           # Main API: read_and_parse_image()
├── __main__.py           # CLI entry point
├── ocr/                  # OCR functionality
│   ├── __init__.py       # Pytesseract wrapper
│   ├── read_word_data.py # Word data class
│   └── read_format_converter.py
└── formatter/            # Text formatting
    ├── __init__.py       # Chord and section parsing
    ├── sheet.py          # Sheet data class
    ├── line.py           # Line data class
    ├── section.py        # Section data class
    └── word.py           # Word data class
```

## See Also

- [Tests](tests/README.md) - Test suite with examples
- [Parser Module](../README.md) - Full parsing pipeline with paper detection
