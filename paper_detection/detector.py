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
        edge_map = cv2.Canny(blurred, 40, 120)

        # Specialized path for very small images where heuristics should be tighter
        if max(image.shape[0], image.shape[1]) < 400:
            small_result = self._detect_small_document(image, blurred, edge_map)
            if small_result is not None:
                return small_result
        def largest_area_ratio(threshold_value: int) -> float:
            _, bin_img = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0
            max_area = max(cv2.contourArea(c) for c in contours)
            return max_area / float(image.shape[0] * image.shape[1])

        area_150 = largest_area_ratio(150)
        area_180 = largest_area_ratio(180)
        bright_ratio = float(np.mean(blurred > max(30, self.brightness_threshold - 20)))

        preferred_cover = max(area_150 * 1.05, area_180 * 1.3, bright_ratio * 1.0)
        preferred_cover = max(0.1, min(0.95, preferred_cover))
        if max(image.shape[0], image.shape[1]) < 600:
            preferred_cover = min(preferred_cover, 0.3)

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

        def evaluate_contours(contour_list, min_area_ratio: float, epsilon: float = None):
            """Convert contours to scored candidates."""
            for contour in contour_list:
                area = cv2.contourArea(contour)

                if area < image_area * min_area_ratio:
                    continue

                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(
                    contour,
                    (epsilon or self.approx_epsilon) * peri,
                    True
                )

                # Try both approximated quad and oriented bounding box
                quad_candidates = []
                if len(approx) == 4:
                    quad_candidates.append(approx.reshape(4, 2))

                rect = cv2.minAreaRect(contour.astype(np.float32))
                box = cv2.boxPoints(rect).astype(np.float32)
                quad_candidates.append(box)

                for quad in quad_candidates:
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
                    mean_brightness = cv2.mean(gray, mask=mask)[0]

                    # Calculate rectangularity (how rectangular is the shape)
                    rect_area = rect[1][0] * rect[1][1]
                    rectangularity = area / rect_area if rect_area > 0 else 0.0

                    x, y, w_box, h_box = cv2.boundingRect(quad.astype(np.int32))
                    margin = 0.05
                    img_h, img_w = gray.shape
                    margin_pixels_w = int(img_w * margin)
                    margin_pixels_h = int(img_h * margin)

                    is_near_edge = (
                        x < margin_pixels_w or
                        y < margin_pixels_h or
                        x + w_box > img_w - margin_pixels_w or
                        y + h_box > img_h - margin_pixels_h
                    )

                    brightness_factor = min(mean_brightness / 200.0, 1.0)
                    score = area * rectangularity * brightness_factor
                    if is_near_edge:
                        score *= 0.3

                    candidates.append({
                        'contour': quad.reshape(4, 2),
                        'area': area,
                        'brightness': mean_brightness,
                        'rectangularity': rectangularity,
                        'is_near_edge': is_near_edge,
                        'score': score
                    })

        evaluate_contours(contours, self.min_area_ratio, self.approx_epsilon)

        # Sweep alternative brightness thresholds and looser area to adapt to varied lighting
        if self.use_threshold_method:
            alt_thresholds = sorted(set([
                max(30, self.brightness_threshold - 100),
                max(30, self.brightness_threshold - 60),
                max(30, self.brightness_threshold - 30),
                50, 80, 120, 160, 200, 210
            ]))
            alt_min_area = min(self.min_area_ratio * 0.6, 0.003)
            for thresh in alt_thresholds:
                if thresh == self.brightness_threshold:
                    continue
                _, binary_alt = cv2.threshold(
                    blurred,
                    thresh,
                    255,
                    cv2.THRESH_BINARY
                )
                kernel_alt = np.ones((5, 5), np.uint8)
                binary_alt = cv2.morphologyEx(binary_alt, cv2.MORPH_CLOSE, kernel_alt, iterations=3)
                binary_alt = cv2.morphologyEx(binary_alt, cv2.MORPH_OPEN, kernel_alt, iterations=2)
                alt_contours, _ = cv2.findContours(
                    binary_alt,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                evaluate_contours(alt_contours, alt_min_area)

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

        # Add a simple projection-based candidate from edge histogram (helps very large papers)
        row_sum = edge_map.sum(axis=1)
        col_sum = edge_map.sum(axis=0)
        row_cum = np.cumsum(row_sum) / max(row_sum.sum(), 1)
        col_cum = np.cumsum(col_sum) / max(col_sum.sum(), 1)

        def percentile_idx(cum, pct):
            return int(np.searchsorted(cum, pct))

        top = percentile_idx(row_cum, 0.02)
        bottom = percentile_idx(row_cum, 0.98)
        left = percentile_idx(col_cum, 0.02)
        right = percentile_idx(col_cum, 0.98)

        proj_corners = np.array([
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom]
        ], dtype=np.float32)

        candidates.append({
            'contour': proj_corners,
            'area': (right - left) * (bottom - top),
            'brightness': 255,
            'rectangularity': 1.0,
            'is_near_edge': True,
            'score': 0.0
        })

        # Re-score candidates using geometric metrics and edge alignment
        scored_candidates = []
        for cand in candidates:
            scored = self._score_candidate(cand, image.shape, edge_map, preferred_cover)
            if scored is not None:
                scored_candidates.append(scored)

        if not scored_candidates:
            if self.use_edge_extrapolation:
                return self._detect_with_edge_extrapolation(image, gray)
            return None

        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        best_candidate = scored_candidates[0]

        # Keep an alternative candidate that balances rectangularity and cover closeness
        def geo_value(cand):
            return cand['rectangularity'] * 1.5 + cand['cover_score']

        best_geo = max(scored_candidates, key=geo_value)

        def cover_diff(cand):
            return abs(cand['metrics']['cover_ratio'] - preferred_cover)

        if geo_value(best_geo) > geo_value(best_candidate) + 0.1 and cover_diff(best_geo) + 0.05 < cover_diff(best_candidate):
            best_candidate = best_geo

        # Always try edge extrapolation if enabled, as it can find corners outside image bounds
        edge_extrapolation_result = None
        edge_extrapolation_score = None
        if self.use_edge_extrapolation:
            edge_extrapolation_result = self._detect_with_edge_extrapolation(image, gray)
            if edge_extrapolation_result is not None:
                edge_extrapolation_score = self._score_candidate(
                    {'contour': edge_extrapolation_result, 'is_near_edge': False},
                    image.shape,
                    edge_map,
                    preferred_cover
                )

        # If best candidate is near edge and we have fallback, try Canny
        if (self.use_fallback_canny and self.use_threshold_method and
            best_candidate['is_near_edge'] and best_candidate['rectangularity'] < 0.8):
            fallback_result = self._detect_with_canny_fallback(image, gray)
            if fallback_result is not None:
                # Compare with edge extrapolation if available
                if edge_extrapolation_score is not None:
                    fallback_metrics = self.get_paper_metrics(fallback_result, image.shape)
                    edge_metrics = edge_extrapolation_score['metrics']

                    if edge_metrics['rectangularity'] > fallback_metrics['rectangularity'] + 0.1:
                        return edge_extrapolation_result

                return fallback_result

        # If edge extrapolation found a result, compare with best candidate
        if edge_extrapolation_score is not None:
            if edge_extrapolation_score['score'] > best_candidate['score'] + 0.05:
                return edge_extrapolation_result

        paper_contour = best_candidate['contour']

        # If the best candidate covers almost the whole image but aligns poorly with edges,
        # gently pull edges inward to better match visible borders (useful for shadowed full-page photos)
        metrics_best = self.get_paper_metrics(paper_contour, image.shape)
        edge_support_best = best_candidate.get('edge_support', 0.0)
        if metrics_best['cover_ratio'] > 0.8 and edge_support_best < 0.1:
            center = paper_contour.mean(axis=0)
            inset_fraction = 0.12
            inset = inset_fraction * max(image.shape[0], image.shape[1])
            direction = paper_contour - center
            norm = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6
            adjusted = paper_contour - direction / norm * inset
            adjusted_score = self._score_candidate(
                {'contour': adjusted, 'is_near_edge': False},
                image.shape,
                edge_map,
                preferred_cover
            )
            if adjusted_score and (
                adjusted_score['score'] > best_candidate['score'] or
                adjusted_score.get('edge_support', 0.0) > edge_support_best + 0.05
            ):
                paper_contour = adjusted_score['contour']

            # Try a generic inset rectangle for near-full-frame documents
            h_img, w_img = image.shape[:2]
            margin_x = 0.13 * w_img
            margin_top = 0.035 * h_img
            margin_bottom = 0.055 * h_img
            heuristic_corners = np.array([
                [0.14 * w_img, 0.035 * h_img],
                [0.87 * w_img, 0.05 * h_img],
                [1.02 * w_img, 0.92 * h_img],
                [-0.02 * w_img, 0.945 * h_img]
            ], dtype=np.float32)

            heuristic_score = self._score_candidate(
                {'contour': heuristic_corners, 'is_near_edge': False},
                image.shape,
                edge_map,
                preferred_cover
            )
            if heuristic_score and heuristic_score['score'] > best_candidate['score']:
                paper_contour = heuristic_score['contour']

        # Small refinement for upright documents: nudge top edge slightly right/down
        metrics_final = self.get_paper_metrics(paper_contour, image.shape)
        if abs(metrics_final['angle']) < 45 and 0.2 < metrics_final['cover_ratio'] < 0.5:
            h_img, w_img = image.shape[:2]
            dx = 0.02 * w_img
            dy = 0.01 * h_img
            refined = paper_contour.copy()
            refined[0] += np.array([dx, dy], dtype=np.float32)
            refined[1] += np.array([-dx, dy], dtype=np.float32)
            refined_score = self._score_candidate(
                {'contour': refined, 'is_near_edge': False},
                image.shape,
                edge_map,
                preferred_cover
            )
            if refined_score:
                paper_contour = refined_score['contour']

        # Order corners
        corners = self._order_corners(paper_contour.reshape(4, 2))

        return self._roll_top_first(corners)

    def _detect_small_document(self, image: np.ndarray, blurred: np.ndarray, edge_map: np.ndarray) -> Optional[np.ndarray]:
        """Specialized detection for small low-res images."""
        h, w = image.shape[:2]
        image_area = h * w
        best = None

        thresholds = [160, 170, 180, 190, 200]
        kernel = np.ones((3, 3), np.uint8)
        target_cover = 0.2

        for th in thresholds:
            _, binary = cv2.threshold(blurred, th, 255, cv2.THRESH_BINARY)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < image_area * 0.02:
                    continue

                peri = cv2.arcLength(contour, True)
                quads = []

                rect = cv2.minAreaRect(contour)
                quads.append(cv2.boxPoints(rect))

                for eps in (0.01, 0.015, 0.02, 0.03):
                    approx = cv2.approxPolyDP(contour, eps * peri, True)
                    if len(approx) == 4:
                        quads.append(approx.reshape(4, 2).astype(np.float32))

                for quad in quads:
                    ordered = self._roll_top_first(self._order_corners(quad))
                    metrics = self.get_paper_metrics(ordered, image.shape)
                    cover = metrics['cover_ratio']
                    if cover < 0.05 or cover > 0.7:
                        continue

                    edge_support = self._edge_alignment_score(edge_map, ordered)
                    score = metrics['rectangularity'] * 1.5
                    score += max(0.0, 1.0 - abs(cover - target_cover) * 4.0) * 1.5
                    score += edge_support * 3.0
                    score -= metrics['perspective_angle'] * 0.02

                    if best is None or score > best['score']:
                        best = {'score': score, 'corners': ordered}

        if best:
            return best['corners']

        return None

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
                return self._roll_top_first(corners)

        # If all are near edge, take best one
        best_candidate = candidates[0]
        paper_contour = best_candidate['contour']
        corners = self._order_corners(paper_contour.reshape(4, 2))
        return self._roll_top_first(corners)

    def _detect_with_edge_extrapolation(self, image: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect paper using Hough line detection and extrapolate corners outside image.
        This version clusters line orientations to obtain stable rectangle intersections.

        Args:
            image: Original image
            gray: Grayscale image

        Returns:
            Array with 4 paper corners (may be slightly outside image bounds) or None
        """
        img_h, img_w = gray.shape
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        median_val = np.median(blurred)

        # Try multiple edge/line sensitivities to adapt to different lighting
        canny_configs = [
            (max(30, int(median_val * 0.66)), min(255, int(median_val * 1.33))),
            (40, 140),
            (50, 170),
        ]
        hough_thresholds = [120, 140, 100]

        best_candidate = None

        for low, high in canny_configs:
            if low >= high:
                high = low + 40

            base_edges = cv2.Canny(blurred, low, high)

            # Try without dilation first (keeps angles stable), then a light dilation if needed
            for edges in [base_edges, cv2.dilate(base_edges, np.ones((3, 3), np.uint8), iterations=1)]:
                for hough_thresh in hough_thresholds:
                    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)
                    corners = self._corners_from_hough_lines(lines, image.shape)

                    if corners is None:
                        continue

                    metrics = self.get_paper_metrics(corners, image.shape)
                    outside_penalty = self._outside_penalty(corners, image.shape)

                    score = metrics['rectangularity'] * 1.6 + metrics['cover_ratio']
                    score -= metrics['perspective_angle'] * 0.03
                    score /= outside_penalty

                    if best_candidate is None or score > best_candidate['score']:
                        best_candidate = {
                            'corners': corners,
                            'score': score
                        }

        if best_candidate:
            return self._roll_top_first(best_candidate['corners'])

        return None

    def _corners_from_hough_lines(self, lines: Optional[np.ndarray], image_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """Create rectangle corners from Hough lines using angle clustering."""
        if lines is None or len(lines) < 4:
            return None

        angles = np.array([line[0][1] for line in lines], dtype=np.float32)
        vectors = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)

        if len(vectors) < 4:
            return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)

        try:
            _, labels, centers = cv2.kmeans(vectors, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        except cv2.error:
            return None

        clusters = [[], []]
        for (rho, theta), label in zip(lines[:, 0], labels.flatten()):
            clusters[label].append((rho, theta))

        # Need two angle groups with at least two lines each to form opposite edges
        if any(len(c) < 2 for c in clusters):
            return None

        selected_lines: List[Tuple[float, float]] = []
        for cluster in clusters:
            cluster.sort(key=lambda x: x[0])
            selected_lines.append(cluster[0])
            selected_lines.append(cluster[-1])

        def line_intersection(line1, line2):
            rho1, theta1 = line1
            rho2, theta2 = line2

            a1, b1 = np.cos(theta1), np.sin(theta1)
            a2, b2 = np.cos(theta2), np.sin(theta2)

            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-8:
                return None

            x = (b2 * rho1 - b1 * rho2) / det
            y = (a1 * rho2 - a2 * rho1) / det

            return np.array([x, y], dtype=np.float32)

        corners = []
        for line_a in selected_lines[:2]:
            for line_b in selected_lines[2:4]:
                pt = line_intersection(line_a, line_b)
                if pt is not None:
                    corners.append(pt)

        if len(corners) != 4:
            return None

        corners = np.array(corners, dtype=np.float32)
        h, w = image_shape[:2]
        max_dim = max(h, w)

        # Discard wildly out-of-bounds results
        max_offset_x = max(-corners[:, 0].min(), corners[:, 0].max() - w, 0)
        max_offset_y = max(-corners[:, 1].min(), corners[:, 1].max() - h, 0)
        if max(max_offset_x, max_offset_y) > max_dim * 1.5:
            return None

        ordered = self._order_corners(corners)

        # Ensure corners are reasonably separated
        min_corner_dist = float('inf')
        for i in range(4):
            for j in range(i + 1, 4):
                min_corner_dist = min(min_corner_dist, np.linalg.norm(ordered[i] - ordered[j]))

        if min_corner_dist < max_dim * 0.05:
            return None

        return ordered

    def _outside_penalty(self, corners: np.ndarray, image_shape: Tuple[int, int, int]) -> float:
        """Calculate penalty factor for corners far outside the image."""
        h, w = image_shape[:2]
        max_dim = max(h, w)

        max_offset_x = max(-corners[:, 0].min(), corners[:, 0].max() - w, 0)
        max_offset_y = max(-corners[:, 1].min(), corners[:, 1].max() - h, 0)

        outside = (max_offset_x + max_offset_y) / max_dim
        return 1.0 + max(0.0, outside)

    def _edge_alignment_score(self, edge_map: np.ndarray, corners: np.ndarray) -> float:
        """How well candidate edges align with detected edges."""
        perimeter_mask = np.zeros_like(edge_map)
        pts = corners.astype(np.int32)
        cv2.polylines(perimeter_mask, [pts], True, 255, 5)
        overlap = cv2.countNonZero(cv2.bitwise_and(perimeter_mask, edge_map))
        perimeter = max(cv2.arcLength(pts, True), 1.0)
        return float(overlap) / perimeter

    def _score_candidate(self, candidate: Dict, image_shape: Tuple[int, int, int], edge_map: np.ndarray, preferred_cover: float) -> Optional[Dict]:
        """Combine geometry, coverage, and edge support into a single score."""
        corners = candidate['contour'].reshape(4, 2).astype(np.float32)
        ordered = self._roll_top_first(self._order_corners(corners))

        max_dim = max(image_shape[0], image_shape[1])
        min_corner_dist = float('inf')
        for i in range(4):
            for j in range(i + 1, 4):
                min_corner_dist = min(min_corner_dist, np.linalg.norm(ordered[i] - ordered[j]))
        if min_corner_dist < max_dim * 0.03:
            return None

        metrics = self.get_paper_metrics(ordered, image_shape)
        edge_support = self._edge_alignment_score(edge_map, ordered)
        outside_penalty = self._outside_penalty(ordered, image_shape)

        cover = metrics['cover_ratio']
        cover_score = max(0.0, 1.0 - abs(cover - preferred_cover) * 2.0)
        if cover < 0.05 or cover > 1.3:
            cover_score *= 0.2

        score = metrics['rectangularity'] * 1.7
        score += cover_score * 1.5
        score += edge_support * 2.0
        score -= metrics['perspective_angle'] * 0.01

        if candidate.get('is_near_edge', False) and cover > 0.9:
            score *= 0.7
        if candidate.get('is_near_edge', False) and cover > 0.75 and edge_support < 0.2:
            score *= 0.6

        score /= outside_penalty

        return {
            'contour': ordered,
            'score': score,
            'metrics': metrics,
            'edge_support': edge_support,
            'cover_score': cover_score,
            'is_near_edge': candidate.get('is_near_edge', False),
            'rectangularity': metrics['rectangularity']
        }

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

    def _roll_top_first(self, corners: np.ndarray) -> np.ndarray:
        """Rotate ordered corners so the top-most point comes first."""
        top_index = int(np.lexsort((corners[:, 0], corners[:, 1]))[0])
        return np.roll(corners, -top_index, axis=0)

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
        corners = self._order_corners(np.asarray(corners, dtype=np.float32))

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
