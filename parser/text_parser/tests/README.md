# Text Parser Tests

Test suite for the `text_parser` module - OCR reading and sheet formatting.

## Test Coverage

- **OCR Reading**: Tests Pytesseract text extraction from cropped sheet images
- **Title Extraction**: Tests both `read_and_parse_image()` and `read_title_only()`
- **Chord Detection**: Verifies chord formatting in output
- **Section Detection**: Verifies section grouping (V1, V2, Chorus, etc.)
- **Error Handling**: Tests invalid paths and arrays
- **Input Formats**: Tests both file paths and numpy arrays

## Test Images

Uses 4 cropped sheet images from `parser/text_parser/tests/images/`:
- `img1.jpg` - Maranatha song sheet (131 KB)
- `img2.jpg` - Modlitba by Ondřej Brzobohatý (1.0 MB)
- `img3.jpg` - Riziková investice song sheet (396 KB)
- `img4.jpg` - Multiple songs including "Jsi dobrý Bůh" (1.9 MB)

## Running Tests

```bash
# From tests directory
cd parser/text_parser/tests
python3 test_text_parser.py

# From project root
python3 -m parser.text_parser.tests.test_text_parser
```

## Test Output Files

Tests automatically save extracted data to `temp/text_parser_output/`:

**Individual Results:**
- `result_<image_name>.json` - Full result with title, data, and image path
- `result_<image_name>.txt` - Human-readable formatted text output

**Combined Results:**
- `all_results.json` - All results combined in single JSON array

All output files use UTF-8 encoding to preserve Czech/Slovak characters.

## Test Results

All tests verify:
- ✓ OCR detects words from images
- ✓ Titles are extracted correctly
- ✓ Data is formatted with chords and sections
- ✓ Functions handle errors gracefully
- ✓ Both path and array inputs work
- ✓ Title consistency between functions
- ✓ Results are saved to temp directory

Total: 11 tests
