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
        use_threshold_method: bool = True,
        use_fallback_canny: bool = True,
        use_edge_extrapolation: bool = True
    ):
        """
        Initialize the detector.

        Args:
            brightness_threshold: Brightness threshold for white paper detection (0-255)
            min_area_ratio: Minimum ratio of paper area to total image
            approx_epsilon: Epsilon for polygon approximation (as ratio of perimeter)
            use_threshold_method: Use threshold method instead of Canny edge detection
            use_fallback_canny: Use Canny edge detection as fallback if threshold method fails
            use_edge_extrapolation: Use Hough line detection to extrapolate corners outside image
        """
        self.brightness_threshold = brightness_threshold
        self.min_area_ratio = min_area_ratio
        self.approx_epsilon = approx_epsilon
        self.use_threshold_method = use_threshold_method
        self.use_fallback_canny = use_fallback_canny
        self.use_edge_extrapolation = use_edge_extrapolation

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

                # Calculate rectangularity (how rectangular is the shape)
                rect = cv2.minAreaRect(approx.astype(np.float32))
                rect_area = rect[1][0] * rect[1][1]
                rectangularity = area / rect_area if rect_area > 0 else 0.0

                # Check if candidate is too close to image edges (likely not paper)
                # Get bounding box
                x, y, w, h = cv2.boundingRect(approx)
                margin = 0.05  # 5% margin from edges
                img_h, img_w = gray.shape
                margin_pixels_w = int(img_w * margin)
                margin_pixels_h = int(img_h * margin)

                is_near_edge = (
                    x < margin_pixels_w or  # Left edge
                    y < margin_pixels_h or  # Top edge
                    x + w > img_w - margin_pixels_w or  # Right edge
                    y + h > img_h - margin_pixels_h  # Bottom edge
                )

                # Brightness factor (normalize, cap at 200 to not over-favor very bright areas)
                brightness_factor = min(mean_brightness / 200.0, 1.0)

                # Score = area * rectangularity * brightness_factor
                # Prefer large, rectangular, reasonably bright areas
                # Penalize candidates near edges
                score = area * rectangularity * brightness_factor
                if is_near_edge:
                    score *= 0.3  # Reduce score for edge candidates

                candidates.append({
                    'contour': approx,
                    'area': area,
                    'brightness': mean_brightness,
                    'rectangularity': rectangularity,
                    'is_near_edge': is_near_edge,
                    'score': score
                })

        if not candidates:
            # Try fallback Canny detection for difficult images
            if self.use_fallback_canny and self.use_threshold_method:
                result = self._detect_with_canny_fallback(image, gray)
                if result is not None:
                    return result

                # If Canny also failed, try edge extrapolation
                if self.use_edge_extrapolation:
                    return self._detect_with_edge_extrapolation(image, gray)

            return None

        # Sort by score (largest and brightest)
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Take the best candidate
        best_candidate = candidates[0]

        # Always try edge extrapolation if enabled, as it can find corners outside image bounds
        edge_extrapolation_result = None
        if self.use_edge_extrapolation:
            edge_extrapolation_result = self._detect_with_edge_extrapolation(image, gray)

        # If best candidate is near edge and we have fallback, try Canny
        if (self.use_fallback_canny and self.use_threshold_method and
            best_candidate['is_near_edge'] and best_candidate['rectangularity'] < 0.8):
            fallback_result = self._detect_with_canny_fallback(image, gray)
            if fallback_result is not None:
                # Compare with edge extrapolation if available
                if edge_extrapolation_result is not None:
                    fallback_metrics = self.get_paper_metrics(fallback_result, image.shape)
                    extrapolation_metrics = self.get_paper_metrics(edge_extrapolation_result, image.shape)

                    # Prefer edge extrapolation if it gives better rectangularity
                    if extrapolation_metrics['rectangularity'] > fallback_metrics['rectangularity'] + 0.1:
                        return edge_extrapolation_result

                return fallback_result

        # If edge extrapolation found a result, compare with best candidate
        if edge_extrapolation_result is not None:
            candidate_corners = best_candidate['contour'].reshape(4, 2)
            candidate_metrics = self.get_paper_metrics(candidate_corners, image.shape)
            edge_metrics = self.get_paper_metrics(edge_extrapolation_result, image.shape)

            # Prefer edge extrapolation if:
            # 1. Rectangularity is significantly better (more rectangular)
            # 2. OR if best candidate is near edge and edge extrapolation has decent rectangularity
            if (edge_metrics['rectangularity'] > candidate_metrics['rectangularity'] + 0.1 or
                (best_candidate['is_near_edge'] and edge_metrics['rectangularity'] > 0.4)):
                return edge_extrapolation_result

        paper_contour = best_candidate['contour']

        # Order corners
        corners = self._order_corners(paper_contour.reshape(4, 2))

        return corners

    def _detect_with_canny_fallback(self, image: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback detection using Canny edge detection with more aggressive parameters.
        Used for difficult images where threshold method fails.

        Args:
            image: Original image
            gray: Grayscale image

        Returns:
            Array with 4 paper corners or None
        """
        # Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection with low thresholds
        edges = cv2.Canny(blurred, 30, 100)

        # More aggressive dilation to connect edges
        kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=3)

        # Find contours
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find quadrilaterals with multiple epsilon values
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.min_area_ratio

        candidates = []
        # Try multiple epsilon values to find best approximation
        epsilon_values = [0.02, 0.025, 0.03, 0.035, 0.04, 0.05]

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            # Polygon approximation - try multiple epsilon values
            peri = cv2.arcLength(contour, True)

            for epsilon_val in epsilon_values:
                approx = cv2.approxPolyDP(contour, epsilon_val * peri, True)

                if len(approx) == 4:
                    # Calculate brightness
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [approx.astype(np.int32)], 255)
                    mean_brightness = cv2.mean(gray, mask=mask)[0]

                    # Calculate rectangularity
                    rect = cv2.minAreaRect(approx.astype(np.float32))
                    rect_area = rect[1][0] * rect[1][1]
                    rectangularity = area / rect_area if rect_area > 0 else 0.0

                    # Check corner distribution - avoid degenerate quads
                    corners = approx.reshape(4, 2)
                    min_corner_dist = float('inf')
                    for i in range(4):
                        for j in range(i+1, 4):
                            dist = np.linalg.norm(corners[i] - corners[j])
                            min_corner_dist = min(min_corner_dist, dist)

                    # Skip if corners are too close (degenerate quad)
                    if min_corner_dist < 50:  # Minimum 50 pixels between corners
                        continue

                    # Check edge proximity
                    x, y, w, h = cv2.boundingRect(approx)
                    margin = 0.05
                    img_h, img_w = gray.shape
                    margin_pixels_w = int(img_w * margin)
                    margin_pixels_h = int(img_h * margin)

                    is_near_edge = (
                        x < margin_pixels_w or
                        y < margin_pixels_h or
                        x + w > img_w - margin_pixels_w or
                        y + h > img_h - margin_pixels_h
                    )

                    # Score prioritizes rectangularity and area, less weight on brightness
                    brightness_factor = min(mean_brightness / 150.0, 1.0)  # Lower threshold for colored paper
                    score = area * rectangularity * brightness_factor
                    if is_near_edge:
                        score *= 0.2  # Heavy penalty for edge candidates

                    # Bonus for good corner distribution
                    corner_dist_factor = min(min_corner_dist / 200.0, 1.0)
                    score *= (0.5 + 0.5 * corner_dist_factor)

                    candidates.append({
                        'contour': approx,
                        'area': area,
                        'brightness': mean_brightness,
                        'rectangularity': rectangularity,
                        'is_near_edge': is_near_edge,
                        'min_corner_dist': min_corner_dist,
                        'score': score
                    })

        if not candidates:
            return None

        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Take best non-edge candidate if possible
        for candidate in candidates:
            if not candidate['is_near_edge']:
                paper_contour = candidate['contour']
                corners = self._order_corners(paper_contour.reshape(4, 2))
                return corners

        # If all are near edge, take best one
        best_candidate = candidates[0]
        paper_contour = best_candidate['contour']
        corners = self._order_corners(paper_contour.reshape(4, 2))
        return corners

    def _detect_with_edge_extrapolation(self, image: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect paper using Hough line detection and extrapolate corners outside image.
        This method can find all 4 corners even if some are outside the image bounds.

        Args:
            image: Original image
            gray: Grayscale image

        Returns:
            Array with 4 paper corners (may be outside image bounds) or None
        """
        # Strong edge detection with stronger parameters to get main edges only
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 75, 200)

        # Detect only strong, long lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=250)  # Higher threshold for main edges only

        if lines is None or len(lines) < 4:
            return None

        # Filter lines on left side of image (where paper typically is)
        img_h, img_w = gray.shape
        left_lines = []

        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0 = a * rho

            # Keep lines on left 70% of image
            if x0 < img_w * 0.7:
                left_lines.append((rho, theta))

        if len(left_lines) < 4:
            return None

        # Find 4 dominant orientations (4 edges of paper)
        # Group lines by angle
        def angle_diff(theta1, theta2):
            diff = abs(theta1 - theta2) * 180 / np.pi
            return min(diff, 180 - diff)

        orientations = []
        for rho, theta in left_lines:
            is_new = True
            for existing_theta in orientations:
                if angle_diff(theta, existing_theta) < 20:  # Within 20 degrees
                    is_new = False
                    break

            if is_new and len(orientations) < 4:
                orientations.append(theta)

        if len(orientations) < 2:  # Need at least 2 orientations (2 pairs of parallel edges)
            return None

        # For each orientation, select both parallel edges (first and last in rho order)
        main_edges = []
        for target_theta in orientations:
            candidates = [(rho, theta) for rho, theta in left_lines
                         if angle_diff(theta, target_theta) < 20]

            if candidates:
                candidates.sort(key=lambda x: x[0])
                # Select first and last (the two parallel edges)
                if len(candidates) >= 2:
                    main_edges.append(candidates[0])
                    main_edges.append(candidates[-1])
                elif len(candidates) == 1:
                    main_edges.append(candidates[0])

        if len(main_edges) < 3:  # Need at least 3 edges to form corners
            return None

        # Calculate intersections of all edge pairs
        def line_intersection(line1, line2):
            rho1, theta1 = line1
            rho2, theta2 = line2

            a1, b1 = np.cos(theta1), np.sin(theta1)
            a2, b2 = np.cos(theta2), np.sin(theta2)

            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-10:
                return None

            x = (b2 * rho1 - b1 * rho2) / det
            y = (a1 * rho2 - a2 * rho1) / det

            return np.array([x, y])

        corners = []
        for i in range(len(main_edges)):
            for j in range(i+1, len(main_edges)):
                intersection = line_intersection(main_edges[i], main_edges[j])
                if intersection is not None:
                    corners.append(intersection)

        if len(corners) < 4:
            return None

        # Remove duplicate corners that are too close together
        corners = np.array(corners)
        unique_corners = []
        for corner in corners:
            is_duplicate = False
            for existing in unique_corners:
                dist = np.linalg.norm(corner - existing)
                if dist < 50:  # Less than 50 pixels apart
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_corners.append(corner)

        corners = np.array(unique_corners)

        if len(corners) < 4:
            return None

        # Find the 4 corners that form the largest quadrilateral
        # Try to select corners that are well-distributed

        # Calculate convex hull to get outer corners
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(corners)
            hull_corners = corners[hull.vertices]

            if len(hull_corners) >= 4:
                # Take first 4 points of convex hull
                corners = hull_corners[:4]
            elif len(hull_corners) == 3:
                # If we only have 3 corners, extrapolate the 4th
                # (this handles the case where one corner is far outside)
                return None
        except:
            # If scipy not available or ConvexHull fails, use simpler method
            # Select 4 corners that are most spread out
            if len(corners) > 4:
                # Calculate center
                center = corners.mean(axis=0)

                # Calculate angles from center
                angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])

                # Sort by angle and pick evenly distributed corners
                sorted_indices = np.argsort(angles)
                step = len(sorted_indices) // 4
                selected_indices = [sorted_indices[i * step] for i in range(4)]
                corners = corners[selected_indices]

        # Order corners properly
        ordered_corners = self._order_corners(corners)

        # Validate that these corners make sense
        # Check that they form a reasonably large quadrilateral
        area = cv2.contourArea(ordered_corners.astype(np.int32))
        image_area = img_h * img_w
        coverage = area / image_area

        # Accept if coverage is reasonable
        # Can be > 1 (even much larger) if corners are outside image bounds
        # Just ensure it's not too small (< 5% of image)
        if abs(coverage) > 0.05:
            return ordered_corners

        return None

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

    def _clip_polygon_to_image(self, corners: np.ndarray, image_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Clip polygon to image boundaries.

        When corners are outside image bounds, this creates a clipped polygon
        that represents only the visible part of the paper.

        Args:
            corners: Original corners (may be outside image)
            image_shape: Shape of the image (height, width, channels)

        Returns:
            Clipped polygon points (may have more than 4 points)
        """
        h, w = image_shape[:2]

        # Create clip rectangle (image boundaries)
        clip_rect = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)

        # Use Sutherland-Hodgman polygon clipping algorithm
        def clip_polygon_component(polygon, edge_p1, edge_p2):
            """Clip polygon against one edge"""
            clipped = []

            for i in range(len(polygon)):
                current = polygon[i]
                next_point = polygon[(i + 1) % len(polygon)]

                # Edge vector
                edge = edge_p2 - edge_p1
                # Normal pointing inward (left of edge)
                normal = np.array([-edge[1], edge[0]])

                # Check if points are inside (left of edge)
                current_inside = np.dot(current - edge_p1, normal) >= 0
                next_inside = np.dot(next_point - edge_p1, normal) >= 0

                if current_inside and next_inside:
                    # Both inside
                    clipped.append(next_point)
                elif current_inside and not next_inside:
                    # Leaving - add intersection
                    t = np.dot(current - edge_p1, normal) / np.dot(current - next_point, normal)
                    intersection = current + t * (next_point - current)
                    clipped.append(intersection)
                elif not current_inside and next_inside:
                    # Entering - add intersection and next
                    t = np.dot(current - edge_p1, normal) / np.dot(current - next_point, normal)
                    intersection = current + t * (next_point - current)
                    clipped.append(intersection)
                    clipped.append(next_point)
                # else: both outside - skip

            return np.array(clipped) if clipped else np.array([])

        # Clip against all 4 edges
        clipped = corners.copy()
        for i in range(4):
            if len(clipped) == 0:
                break
            edge_p1 = clip_rect[i]
            edge_p2 = clip_rect[(i + 1) % 4]
            clipped = clip_polygon_component(clipped, edge_p1, edge_p2)

        return clipped

    def get_paper_metrics(self, corners: np.ndarray, image_shape: Tuple[int, int, int]) -> Dict[str, float]:
        """
        Calculate quality metrics for detected paper.

        Args:
            corners: Ordered paper corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    (corners may be outside image bounds)
            image_shape: Shape of the original image (height, width, channels)

        Returns:
            Dictionary with metrics:
                - cover_ratio: Ratio of VISIBLE paper area to image area (0-1)
                - rectangularity: How rectangular the shape is (0-1, 1=perfect rectangle)
                - angle: Rotation angle of paper relative to horizontal (-90 to 90 degrees)
                - perspective_angle: Perspective distortion angle (0=no distortion)
        """
        # Clip polygon to image boundaries for accurate cover_ratio
        clipped_corners = self._clip_polygon_to_image(corners, image_shape)

        # Calculate VISIBLE area (clipped to image boundaries)
        if len(clipped_corners) >= 3:
            visible_area = cv2.contourArea(clipped_corners.astype(np.int32))
        else:
            visible_area = 0

        # Calculate image area
        image_area = image_shape[0] * image_shape[1]

        # 1. Cover ratio - what percentage of image does the VISIBLE paper occupy
        cover_ratio = visible_area / image_area if image_area > 0 else 0

        # 2. Rectangularity - how well does the FULL paper shape fit a rectangle
        # Use original corners (even if outside image) for geometric properties
        full_paper_area = abs(cv2.contourArea(corners.astype(np.float32)))

        # Get minimum area rectangle of full paper
        rect = cv2.minAreaRect(corners.astype(np.float32))
        rect_area = rect[1][0] * rect[1][1]

        if rect_area > 0 and full_paper_area > 0:
            rectangularity = full_paper_area / rect_area
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
