# Paper Transform Tests

## Test Files

### `test_warp.py`
Unit tests for warp_paper function:
- Corner ordering
- Dimension calculation
- Basic transformation logic

### `test_warp_integration.py`
Integration tests with real images:
- Tests warp transformation on 3 real photos stored locally in `test_data/`
- Compares results against reference images
- Validates consistency and accuracy

**Test Images:**
1. `IMG_20230826_092429.jpg` - Straight overhead shot (95.8% similarity)
2. `IMG_20230826_093159.jpg` - Angled shot (93.9% similarity)
3. `test_image_2.jpg` - Test image (100% similarity)

**Test Data Structure:**
```
test_data/
├── IMG_20230826_092429.jpg      # Source images
├── IMG_20230826_093159.jpg
├── test_image_2.jpg
├── test_corners.json            # Corner coordinates
└── warp_references/             # Expected outputs
    ├── IMG_20230826_092429_warped.jpg
    ├── IMG_20230826_093159_warped.jpg
    └── test_image_2_warped.jpg
```

**How it works:**
- Loads images and corners from `test_data/`
- Applies `warp_paper()` transformation
- Compares with reference images using MSE (Mean Squared Error)
- Validates similarity > 93%

## Running Tests

```bash
# Run all warp tests
pytest paper_transform/tests/test_warp.py -v

# Run integration tests
pytest paper_transform/tests/test_warp_integration.py -v

# Run all tests
pytest paper_transform/tests/ -v
```

## Regenerating References

If you update the warp algorithm, regenerate reference images:

```bash
python3 temp/generate_warp_references.py
```

This will update the reference images in `test_data/warp_references/`.
