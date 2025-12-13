# Parser Module

Document processing pipeline - paper detection, transformation, and parsing.

## Current Status

The module is in development. Currently available: end-to-end pipeline tests demonstrating paper detection and transformation.

## Testing

### Run Pipeline Tests

Tests demonstrate the complete workflow: detect paper corners → transform perspective → save visualizations.

```bash
# Run all tests
pytest parser/tests/test_detection_and_transform.py -v

# Run specific tests
pytest parser/tests/test_detection_and_transform.py::TestDetectionAndTransform::test_detect_and_transform_all -v -s
pytest parser/tests/test_detection_and_transform.py::TestDetectionAndTransform::test_detect_and_transform_sample -v -s
pytest parser/tests/test_detection_and_transform.py::TestDetectionAndTransform::test_pipeline_statistics -v -s
```

### Test Outputs

All visualizations are saved to `parser/tests/output/`:

**Pipeline visualizations** (`pipeline_*.jpg`):
- Side-by-side comparison: Original with detected corners | Transformed paper
- Generated for all 93 test images

**Detailed visualizations** (`detailed_*.jpg`):
- 3-column view: Original | Original with corners | Transformed
- Generated for first 5 sample images

### Test Results Summary

```
Total images: 93
Detection success rate: 100.0%
Transformation success rate: 100.0%

Output sizes:
  Height: 127 - 3622 px (avg: 1993 px)
  Width: 77 - 3008 px (avg: 1737 px)
```

## Module Structure

```
parser/
├── __init__.py           # Module initialization
├── README.md            # This file
└── tests/
    ├── __init__.py
    ├── test_detection_and_transform.py  # End-to-end pipeline tests
    └── output/          # Test visualizations (gitignored)
```

## Dependencies

- `paper_detection` - Paper corner detection using U-Net segmentation
- `paper_transform` - Perspective transformation and orientation correction

## Future Development

- Text extraction (OCR)
- Content parsing and structuring
- Export to various formats
