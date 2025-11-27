"""
Paper detector for images using OpenCV
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict


class PaperDetector:
    """
    Class for paper detection in images.

    Uses edge detection and contour algorithms to find
    rectangular paper in the image.
    """

    def __init__(
        self,
        brightness_threshold: int = 200,
        min_area_ratio: float = 0.01,
        approx_epsilon: float = 0.02,
        use_threshold_method: bool = True
    ):
        """
        Initialize the detector.

        Args:
            brightness_threshold: Brightness threshold for white paper detection (0-255)
            min_area_ratio: Minimum ratio of paper area to total image
            approx_epsilon: Epsilon for polygon approximation (as ratio of perimeter)
            use_threshold_method: Use threshold method instead of Canny edge detection
        """
        self.brightness_threshold = brightness_threshold
        self.min_area_ratio = min_area_ratio
        self.approx_epsilon = approx_epsilon
        self.use_threshold_method = use_threshold_method

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect paper in the image.

        Args:
            image: Input image (BGR format)

        Returns:
            Array with 4 paper corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            or None if paper was not found.
            Corners are ordered: top-left, top-right, bottom-right, bottom-left
        """
        if image is None or image.size == 0:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.use_threshold_method:
            # Threshold-based method - detect white areas
            _, binary = cv2.threshold(
                blurred,
                self.brightness_threshold,
                255,
                cv2.THRESH_BINARY
            )

            # Morphological operations for cleanup
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
        else:
            # Older method based on Canny edge detection
            # (kept for compatibility)
            edges = cv2.Canny(blurred, 50, 150)

            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)

            contours, _ = cv2.findContours(
                dilated,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

        if not contours:
            return None

        # Minimum paper area
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.min_area_ratio

        # Find quadrilaterals and evaluate by brightness (white paper)
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Skip too small contours
            if area < min_area:
                continue

            # Polygon approximation
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(
                contour,
                self.approx_epsilon * peri,
                True
            )

            # Looking for quadrilateral (4 corners)
            if len(approx) == 4:
                # Calculate average brightness inside contour
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [approx.astype(np.int32)], 255)
                mean_brightness = cv2.mean(gray, mask=mask)[0]

                # Score = area * brightness (prefer large and bright areas)
                score = area * (mean_brightness / 255.0)

                candidates.append({
                    'contour': approx,
                    'area': area,
                    'brightness': mean_brightness,
                    'score': score
                })

        if not candidates:
            return None

        # Sort by score (largest and brightest)
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Take the best candidate
        best_candidate = candidates[0]
        paper_contour = best_candidate['contour']

        # Order corners
        corners = self._order_corners(paper_contour.reshape(4, 2))

        return corners

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners in correct order: top-left, top-right, bottom-right, bottom-left.

        Args:
            corners: Array with 4 corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            Ordered corners
        """
        # Calculate coordinates relative to centroid
        center = corners.mean(axis=0)

        # Calculate angles from centroid
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])

        # Sort by angles
        # Top-left has angle around -135° (-3π/4)
        # Top-right has angle around -45° (-π/4)
        # Bottom-right has angle around 45° (π/4)
        # Bottom-left has angle around 135° (3π/4)

        ordered_indices = []

        # Top-left: angle in range [-π, -π/2]
        tl_idx = np.argmin(angles)
        ordered_indices.append(tl_idx)

        # Remove already used corner
        remaining_corners = [(i, angles[i]) for i in range(4) if i != tl_idx]

        # Top-right: smallest angle from remaining (will be in range [-π/2, 0])
        remaining_corners.sort(key=lambda x: x[1])

        # Determine top-right, bottom-right, bottom-left
        # by y-coordinate and angle
        remaining_corners_with_y = [
            (idx, angle, corners[idx][1])
            for idx, angle in remaining_corners
        ]

        # Top-right: smallest y from remaining
        tr_candidates = [(idx, angle, y) for idx, angle, y in remaining_corners_with_y]
        tr_candidates.sort(key=lambda x: x[2])
        tr_idx = tr_candidates[0][0]
        ordered_indices.append(tr_idx)

        # Bottom-right and bottom-left
        remaining = [idx for idx, _, _ in tr_candidates[1:]]

        # Bottom-right: largest x from remaining two
        br_candidates = [(idx, corners[idx][0]) for idx in remaining]
        br_candidates.sort(key=lambda x: x[1], reverse=True)
        br_idx = br_candidates[0][0]
        bl_idx = br_candidates[1][0]

        ordered_indices.append(br_idx)
        ordered_indices.append(bl_idx)

        return corners[ordered_indices]

    def get_paper_dimensions(self, corners: np.ndarray) -> Tuple[int, int]:
        """
        Calculate paper dimensions from corners.

        Args:
            corners: Ordered paper corners

        Returns:
            Tuple (width, height) in pixels
        """
        # Calculate width (average of top and bottom edges)
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        width = int(max(width_top, width_bottom))

        # Calculate height (average of left and right edges)
        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        height = int(max(height_left, height_right))

        return width, height

    def get_paper_metrics(self, corners: np.ndarray, image_shape: Tuple[int, int, int]) -> Dict[str, float]:
        """
        Calculate quality metrics for detected paper.

        Args:
            corners: Ordered paper corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            image_shape: Shape of the original image (height, width, channels)

        Returns:
            Dictionary with metrics:
                - cover_ratio: Ratio of paper area to image area (0-1)
                - rectangularity: How rectangular the shape is (0-1, 1=perfect rectangle)
                - angle: Rotation angle of paper relative to horizontal (-90 to 90 degrees)
                - perspective_angle: Perspective distortion angle (0=no distortion)
        """
        # Calculate contour area
        contour_area = cv2.contourArea(corners.astype(np.int32))

        # Calculate image area
        image_area = image_shape[0] * image_shape[1]

        # 1. Cover ratio - what percentage of image does the paper occupy
        cover_ratio = contour_area / image_area

        # 2. Rectangularity - how well does the contour fit a rectangle
        # Get minimum area rectangle
        rect = cv2.minAreaRect(corners.astype(np.float32))
        rect_area = rect[1][0] * rect[1][1]

        if rect_area > 0:
            rectangularity = contour_area / rect_area
        else:
            rectangularity = 0.0

        # 3. Angle - rotation angle of the paper
        # From minAreaRect: (center, (width, height), angle)
        # OpenCV returns angle in range [-90, 0)
        angle = rect[2]

        # Normalize angle to [-90, 90] range
        # If width < height, the rectangle is vertical, adjust angle
        if rect[1][0] < rect[1][1]:
            angle = 90 + angle

        # 4. Perspective angle - measure of perspective distortion
        # Calculate by comparing opposite edge lengths
        # Ordered corners: top-left, top-right, bottom-right, bottom-left

        # Top edge length
        top_length = np.linalg.norm(corners[1] - corners[0])
        # Bottom edge length
        bottom_length = np.linalg.norm(corners[2] - corners[3])
        # Left edge length
        left_length = np.linalg.norm(corners[3] - corners[0])
        # Right edge length
        right_length = np.linalg.norm(corners[2] - corners[1])

        # Calculate perspective distortion ratios
        horizontal_distortion = abs(top_length - bottom_length) / max(top_length, bottom_length)
        vertical_distortion = abs(left_length - right_length) / max(left_length, right_length)

        # Perspective angle as average distortion converted to degrees
        # 0% distortion = 0°, 100% distortion ≈ 45° (severe perspective)
        avg_distortion = (horizontal_distortion + vertical_distortion) / 2
        perspective_angle = np.arctan(avg_distortion) * 180 / np.pi

        return {
            'cover_ratio': float(cover_ratio),
            'rectangularity': float(rectangularity),
            'angle': float(angle),
            'perspective_angle': float(perspective_angle)
        }
