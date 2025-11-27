# Test Images

This directory contains test images for the paper detection module.

## Test Images

### test_image_1.jpeg
- **Type**: Small paper on wooden floor
- **Characteristics**: Small, angled paper
- **Detection parameters**: `brightness_threshold=180`
- **Dimensions**: 217x232 px

### test_image_2.jpg
- **Type**: Full-size test/exam paper
- **Characteristics**: Large paper with text, slight shadow
- **Detection parameters**: `brightness_threshold=100`
- **Dimensions**: 3024x4032 px

### test_image_3.jpeg
- **Type**: Document/invoice on dark surface
- **Characteristics**: Medium-sized paper on darker background
- **Detection parameters**: `brightness_threshold=120` or Canny edge detection
- **Dimensions**: 1500x880 px

## Usage in Tests

These images are used in parametrized tests to ensure the paper detection module works across various:
- Paper sizes (small to large)
- Lighting conditions (bright to shadowed)
- Background contrasts (light wood to dark surfaces)
- Paper orientations (straight and angled)

The tests use adaptive detection strategies to handle different paper characteristics automatically.
