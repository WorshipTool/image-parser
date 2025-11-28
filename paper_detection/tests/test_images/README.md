# Test Images

This directory contains test images for the paper detection module.

## Test Images

### IMG_20230826_092914.jpg
- **Type**: Small yellow/beige paper held in hand
- **Characteristics**: Colored paper (not white), strong perspective distortion, challenging lighting
- **Detection parameters**: `brightness_threshold=180, min_area_ratio=0.005` with Canny fallback
- **Dimensions**: 2736x3648 px
- **Expected metrics**:
  - Cover ratio: 9.9%
  - Rectangularity: 49.4%
  - Angle: 59.5°
  - Perspective: 27.7°
- **Note**: Uses fallback Canny edge detection due to colored paper and bright window in background

### IMG_20230826_093159.jpg
- **Type**: Standard paper document
- **Characteristics**: Well-lit, good contrast, minimal perspective
- **Detection parameters**: `brightness_threshold=200, min_area_ratio=0.01`
- **Dimensions**: 2736x3648 px
- **Expected metrics**:
  - Cover ratio: 25.0%
  - Rectangularity: 97.3%
  - Angle: 19.9°
  - Perspective: 1.6°

### test_image_1.jpeg
- **Type**: Small paper on wooden floor
- **Characteristics**: Small, angled paper, moderate perspective
- **Detection parameters**: `brightness_threshold=180, min_area_ratio=0.005`
- **Dimensions**: 217x232 px
- **Expected metrics**:
  - Cover ratio: 18.0%
  - Rectangularity: 87.6%
  - Angle: 107.5°
  - Perspective: 6.7°

### test_image_2.jpg
- **Type**: Full-size test/exam paper
- **Characteristics**: Large paper with text, significant shadows
- **Detection parameters**: `brightness_threshold=50, min_area_ratio=0.003`
- **Dimensions**: 3024x4032 px
- **Expected metrics**:
  - Cover ratio: 94.0%
  - Rectangularity: 94.0%
  - Angle: 90.0°
  - Perspective: 0.5°

### test_image_3.jpeg
- **Type**: Document/invoice on dark surface
- **Characteristics**: Medium-sized paper on darker background
- **Detection parameters**: `brightness_threshold=120, min_area_ratio=0.003`
- **Dimensions**: 1500x880 px
- **Expected metrics**:
  - Cover ratio: 47.2%
  - Rectangularity: 88.0%
  - Angle: 177.5°
  - Perspective: 6.7°

## Usage in Tests

These images are used in parametrized tests to ensure the paper detection module works across various:
- Paper sizes (9.9% to 94% cover ratio)
- Paper colors (white to yellow/beige)
- Lighting conditions (bright to heavily shadowed)
- Background contrasts (light wood to dark surfaces, bright windows)
- Paper orientations (19.9° to 177.5° rotation)
- Perspective distortion (0.5° to 27.7°)
- Rectangularity (49.4% to 97%)

The tests use adaptive detection strategies to handle different paper characteristics automatically.

## Metrics Tests

The `test_metrics_ranges.py` file contains parametrized tests that verify each image produces metrics within expected ranges:
- All 4 metrics must be within tolerance
- Individual metric range tests for each metric
- Visualization tests with metrics overlay
