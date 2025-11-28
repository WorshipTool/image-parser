#!/usr/bin/env python3
"""Debug corner detection for IMG_20230826_092914.jpg"""

import cv2
import numpy as np
from paper_detection import PaperDetector

image_path = "paper_detection/tests/test_images/IMG_20230826_092914.jpg"
image = cv2.imread(image_path)

print(f"Image size: {image.shape[1]}x{image.shape[0]}")

# Try different detection methods
detector = PaperDetector(brightness_threshold=180, min_area_ratio=0.005)

# Test threshold method
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("\n=== Testing threshold method ===")
threshold_corners = detector._detect_with_threshold(image, gray)
if threshold_corners is not None:
    print("Threshold method corners:")
    for i, corner in enumerate(threshold_corners):
        in_img = 0 <= corner[0] < image.shape[1] and 0 <= corner[1] < image.shape[0]
        print(f"  {i}: ({corner[0]:8.1f}, {corner[1]:8.1f}) {'✓' if in_img else '✗'}")

print("\n=== Testing Canny fallback ===")
canny_corners = detector._detect_with_canny_fallback(image, gray)
if canny_corners is not None:
    print("Canny fallback corners:")
    for i, corner in enumerate(canny_corners):
        in_img = 0 <= corner[0] < image.shape[1] and 0 <= corner[1] < image.shape[0]
        print(f"  {i}: ({corner[0]:8.1f}, {corner[1]:8.1f}) {'✓' if in_img else '✗'}")

print("\n=== Testing edge extrapolation ===")
edge_corners = detector._detect_with_edge_extrapolation(image, gray)
if edge_corners is not None:
    print("Edge extrapolation corners:")
    for i, corner in enumerate(edge_corners):
        in_img = 0 <= corner[0] < image.shape[1] and 0 <= corner[1] < image.shape[0]
        print(f"  {i}: ({corner[0]:8.1f}, {corner[1]:8.1f}) {'✓' if in_img else '✗'}")

print("\n=== Full detect() pipeline ===")
final_corners = detector.detect(image)
if final_corners is not None:
    print("Final corners:")
    for i, corner in enumerate(final_corners):
        in_img = 0 <= corner[0] < image.shape[1] and 0 <= corner[1] < image.shape[0]
        print(f"  {i}: ({corner[0]:8.1f}, {corner[1]:8.1f}) {'✓' if in_img else '✗'}")
