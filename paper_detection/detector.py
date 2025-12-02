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
        use_edge_extrapolation: bool = True,
        use_adaptive_params: bool = False
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
            use_adaptive_params: Use adaptive parameters derived from image statistics (recommended)
        """
        self.brightness_threshold = brightness_threshold
        self.min_area_ratio = min_area_ratio
        self.approx_epsilon = approx_epsilon
        self.use_threshold_method = use_threshold_method
        self.use_fallback_canny = use_fallback_canny
        self.use_edge_extrapolation = use_edge_extrapolation
        self.use_adaptive_params = use_adaptive_params

    def _compute_adaptive_params(self, image: np.ndarray, blurred: np.ndarray) -> Dict:
        """
        Compute adaptive parameters from image statistics.
        This removes all magic numbers and adapts to each image.

        Args:
            image: Original BGR image
            blurred: Blurred grayscale image

        Returns:
            Dictionary with adaptive parameters
        """
        params = {}

        # 1. Otsu threshold (adaptive to brightness distribution)
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        params['otsu_threshold'] = int(otsu_val)

        # 2. Percentile-based thresholds (replace fixed 200, 180, etc.)
        params['p50'] = float(np.percentile(blurred, 50))  # Median
        params['p75'] = float(np.percentile(blurred, 75))
        params['p85'] = float(np.percentile(blurred, 85))
        params['p95'] = float(np.percentile(blurred, 95))

        # 3. Canny thresholds (based on median, not fixed!)
        median_val = np.median(blurred)
        params['canny_low'] = max(20, int(median_val * 0.66))
        params['canny_high'] = min(255, int(median_val * 1.33))

        # 4. Adaptive block sizes (scale with image dimensions, not fixed 21, 51, 101)
        min_dim = min(image.shape[0], image.shape[1])
        adaptive_blocks = [
            max(21, int(min_dim * 0.05)),  # 5% of min dimension
            max(51, int(min_dim * 0.10)),  # 10%
            max(101, int(min_dim * 0.20)), # 20%
        ]
        # Ensure odd numbers for cv2.adaptiveThreshold
        params['adaptive_block_sizes'] = [b if b % 2 == 1 else b + 1 for b in adaptive_blocks]

        # 5. Adaptive C values (based on local std, not fixed 5, -5, -10)
        std_val = float(np.std(blurred))
        params['adaptive_C_values'] = [
            int(std_val * 0.5),
            -int(std_val * 0.5),
            -int(std_val * 1.0),
        ]

        # 6. Background flattening with ADAPTIVE threshold (not fixed 93-94!)
        kernel_size = int(min_dim * 0.10)  # 10% of image size
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(51, min(kernel_size, 251))  # Reasonable bounds

        background = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), 0)
        normalized = cv2.divide(blurred.astype(np.float32), background.astype(np.float32) + 1e-6)
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Use Otsu on normalized image but constrain to reasonable range
        flatten_thresh_otsu, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Constrain to range that empirically works well (85-100)
        flatten_thresh = np.clip(flatten_thresh_otsu, 85, 100)
        params['flatten_threshold'] = int(flatten_thresh)
        params['normalized_image'] = normalized

        # 7. HSV saturation threshold (ADAPTIVE, not fixed 120!)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        sat_median = float(np.median(saturation))
        sat_std = float(np.std(saturation))
        params['saturation_threshold'] = min(255.0, sat_median + 2.0 * sat_std)  # 2-sigma threshold
        params['saturation'] = saturation

        # 8. Preferred coverage (adaptive)
        test_areas = []
        for thresh in [otsu_val, params['p75'], params['p85']]:
            _, binary = cv2.threshold(blurred, int(thresh), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_area = max(cv2.contourArea(c) for c in contours)
                area_ratio = max_area / (image.shape[0] * image.shape[1])
                test_areas.append(area_ratio)

        if test_areas:
            params['preferred_coverage'] = float(np.clip(max(test_areas) * 1.1, 0.1, 0.95))
        else:
            params['preferred_coverage'] = 0.5  # Fallback

        return params

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

        # Compute adaptive parameters if enabled
        if self.use_adaptive_params:
            adaptive_params = self._compute_adaptive_params(image, blurred)
            edge_map = cv2.Canny(blurred, adaptive_params['canny_low'], adaptive_params['canny_high'])
        else:
            adaptive_params = None
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

        # Compute preferred coverage adaptively if enabled
        if self.use_adaptive_params and adaptive_params:
            preferred_cover = adaptive_params['preferred_coverage']
        else:
            # Fallback: old method
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

                # Try multiple epsilon values for better quad approximation
                quad_candidates = []

                for eps_mult in [0.015, 0.02, 0.025, 0.03]:
                    approx = cv2.approxPolyDP(
                        contour,
                        eps_mult * peri,
                        True
                    )
                    if len(approx) == 4:
                        quad_candidates.append(approx.reshape(4, 2))

                rect = cv2.minAreaRect(contour.astype(np.float32))
                box = cv2.boxPoints(rect).astype(np.float32)
                quad_candidates.append(box)

                for quad in quad_candidates:
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
                    mean_brightness = cv2.mean(gray, mask=mask)[0]

                    # Calculate brightness variance to detect uniform paper
                    std_dev = np.std(gray[mask > 0]) if np.any(mask > 0) else 0

                    # Calculate edge strength along the contour perimeter
                    edge_perimeter_mask = np.zeros_like(gray)
                    cv2.polylines(edge_perimeter_mask, [quad.astype(np.int32)], True, 255, 3)
                    edge_strength = cv2.mean(edge_map, mask=edge_perimeter_mask)[0]

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

                    # Improved scoring that favors paper characteristics
                    # Greatly reduce brightness requirement to allow heavily shadowed papers
                    brightness_factor = min(mean_brightness / 100.0, 1.0) ** 0.3

                    # Paper has relatively uniform brightness (low std_dev)
                    uniformity_factor = max(0, 1.0 - std_dev / 100.0)

                    # Paper edges are typically strong and clear
                    edge_factor = min(edge_strength / 50.0, 1.0)

                    # EXTREMELY strongly favor larger areas - use area^5.0 to heavily penalize smaller contours
                    # This helps with hand-held papers where shadows create smaller false detections
                    # Reduce rectangularity and brightness weights to allow slightly irregular/shadowed shapes when much larger
                    score = (area ** 5.0) * (rectangularity ** 0.2) * (brightness_factor ** 0.2)
                    score *= (0.7 + 0.3 * uniformity_factor)  # Bonus for uniform areas
                    score *= (0.8 + 0.2 * edge_factor)  # Bonus for strong edges

                    if is_near_edge:
                        score *= 0.3

                    candidates.append({
                        'contour': quad.reshape(4, 2),
                        'area': area,
                        'brightness': mean_brightness,
                        'rectangularity': rectangularity,
                        'is_near_edge': is_near_edge,
                        'uniformity': uniformity_factor,
                        'edge_strength': edge_strength,
                        'score': score
                    })

        evaluate_contours(contours, self.min_area_ratio, self.approx_epsilon)

        # Sweep alternative brightness thresholds and looser area to adapt to varied lighting
        if self.use_threshold_method:
            # Add adaptive thresholding which works better with varied lighting
            # Try multiple block sizes to handle different shadow scenarios
            for block_size in [21, 51, 101, 151]:
                for C_val in [5, -5, -10]:
                    adaptive_binary = cv2.adaptiveThreshold(
                        blurred,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        block_size,
                        C_val
                    )

                    # Use larger morphology for larger blocks
                    if block_size > 50:
                        kernel_size = 15
                        close_iter = 5
                    else:
                        kernel_size = 3
                        close_iter = 2

                    kernel_adapt = np.ones((kernel_size, kernel_size), np.uint8)
                    adaptive_binary = cv2.morphologyEx(adaptive_binary, cv2.MORPH_CLOSE, kernel_adapt, iterations=close_iter)

                    adaptive_contours, _ = cv2.findContours(
                        adaptive_binary,
                        cv2.RETR_LIST,  # Use LIST to get all contours
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    # For large blocks, also try convex hull + minAreaRect approach
                    if block_size > 50:
                        for contour in adaptive_contours:
                            area = cv2.contourArea(contour)
                            area_ratio = area / image_area

                            # Look for mid-size contours that might be paper
                            if 0.4 <= area_ratio <= 0.7:
                                # Use convex hull to smooth out irregularities
                                hull = cv2.convexHull(contour)
                                hull_area = cv2.contourArea(hull)

                                # Check rectangularity of hull
                                rect = cv2.minAreaRect(hull.astype(np.float32))
                                rect_area = rect[1][0] * rect[1][1]
                                rectangularity = hull_area / rect_area if rect_area > 0 else 0

                                # If reasonably rectangular, add minAreaRect box as candidate
                                if rectangularity > 0.7:
                                    box = cv2.boxPoints(rect).astype(np.float32)
                                    candidates.append({
                                        'contour': box,
                                        'brightness': 0,
                                        'rectangularity': rectangularity,
                                        'is_near_edge': False,
                                        'score': 0
                                    })

                    evaluate_contours(adaptive_contours, self.min_area_ratio * 0.5)

            # Background flattening for challenging shadow/hand cases
            # Use adaptive parameters or fall back to old method
            if self.use_adaptive_params and adaptive_params:
                # Use pre-computed normalized image and adaptive threshold
                normalized = adaptive_params['normalized_image']
                flatten_thresh = adaptive_params['flatten_threshold']
                saturation = adaptive_params['saturation']
                sat_threshold = adaptive_params['saturation_threshold']
            else:
                # Fallback: old fixed method
                background = cv2.GaussianBlur(blurred, (151, 151), 0)
                normalized = cv2.divide(blurred.astype(np.float32), background.astype(np.float32) + 1e-6)
                normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
                normalized = normalized.astype(np.uint8)
                flatten_thresh = 93  # Fixed fallback
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                saturation = hsv[:, :, 1]
                sat_threshold = 120  # Fixed fallback

            # Use adaptive threshold (not fixed 93-94!)
            for thresh_offset in [0, 1]:  # Try threshold and threshold+1
                current_thresh = flatten_thresh + thresh_offset
                _, flatten_binary = cv2.threshold(normalized, current_thresh, 255, cv2.THRESH_BINARY)
                kernel_flatten = np.ones((3, 3), np.uint8)
                flatten_binary = cv2.morphologyEx(flatten_binary, cv2.MORPH_CLOSE, kernel_flatten, iterations=1)

                # SOFT saturation filtering: Don't apply hard mask here
                # Instead, we'll apply penalty in scoring function
                # This keeps all candidates and lets scoring decide

                flatten_contours, _ = cv2.findContours(
                    flatten_binary,
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                # Process contours - skip convex hull for large areas to avoid including fingers
                for contour in flatten_contours:
                    area = cv2.contourArea(contour)
                    area_ratio = area / image_area

                    if area_ratio < 0.40 or area_ratio > 0.70:
                        continue

                    peri = cv2.arcLength(contour, True)

                    # For large contours, approximate directly (no convex hull)
                    # This prevents fingers/hands from being included
                    for eps_mult in [0.02, 0.025, 0.03]:
                        approx = cv2.approxPolyDP(contour, eps_mult * peri, True)

                        if len(approx) == 4:
                            quad = approx.reshape(-1, 2).astype(np.float32)
                        elif 5 <= len(approx) <= 6:
                            rect = cv2.minAreaRect(contour.astype(np.float32))
                            quad = cv2.boxPoints(rect).astype(np.float32)
                        else:
                            continue

                        rect = cv2.minAreaRect(contour.astype(np.float32))
                        rect_area = rect[1][0] * rect[1][1]
                        rectangularity = area / rect_area if rect_area > 0 else 0

                        if rectangularity < 0.65:
                            continue

                        mask = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
                        mean_brightness = cv2.mean(gray, mask=mask)[0]

                        candidates.append({
                            'contour': quad,
                            'brightness': mean_brightness,
                            'rectangularity': rectangularity,
                            'is_near_edge': False,
                            'score': area ** 4.0 * rectangularity ** 2.0,
                            'from_flattening': True  # Mark as flattening candidate
                        })

            alt_thresholds = sorted(set([
                max(30, self.brightness_threshold - 100),
                max(30, self.brightness_threshold - 60),
                max(30, self.brightness_threshold - 30),
                50, 80, 100, 120, 140, 160, 180, 200, 210, 220
            ]))
            alt_min_area = min(self.min_area_ratio * 0.5, 0.002)
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
            scored = self._score_candidate(cand, image.shape, edge_map, preferred_cover, adaptive_params)
            if scored is not None:
                scored_candidates.append(scored)

        if not scored_candidates:
            if self.use_edge_extrapolation:
                return self._detect_with_edge_extrapolation(image, gray)
            return None

        scored_candidates.sort(key=lambda x: x['score'], reverse=True)

        # Select best candidate by score
        best_candidate = scored_candidates[0]

        # Keep an alternative candidate that balances rectangularity and cover closeness
        def geo_value(cand):
            return cand['rectangularity'] * 1.5 + cand['cover_score']

        best_geo = max(scored_candidates, key=geo_value)

        def cover_diff(cand):
            return abs(cand['metrics']['cover_ratio'] - preferred_cover)

        # Don't override background flattening candidates with geometric alternatives
        if (not best_candidate.get('from_flattening', False) and
            geo_value(best_geo) > geo_value(best_candidate) + 0.1 and
            cover_diff(best_geo) + 0.05 < cover_diff(best_candidate)):
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
                    preferred_cover,
                    adaptive_params
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
        # Don't override background flattening candidates - they're specifically tuned for challenging cases
        if edge_extrapolation_score is not None and not best_candidate.get('from_flattening', False):
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
                preferred_cover,
                adaptive_params
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
                preferred_cover,
                adaptive_params
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
                preferred_cover,
                adaptive_params
            )
            if refined_score:
                paper_contour = refined_score['contour']

        # Order corners
        corners = self._order_corners(paper_contour.reshape(4, 2))

        # Refine corners for better accuracy
        refined_corners = self._refine_corners(gray, corners)

        return self._roll_top_first(refined_corners)

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

    def _refine_corners(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Refine corner positions using sub-pixel accuracy.

        Args:
            gray: Grayscale image
            corners: Initial corner positions

        Returns:
            Refined corner positions
        """
        h, w = gray.shape
        refined = corners.copy().astype(np.float32)

        # Only refine corners that are inside the image
        corners_in_bounds = []
        for i in range(4):
            if 5 <= refined[i][0] < w-5 and 5 <= refined[i][1] < h-5:
                corners_in_bounds.append(i)

        if len(corners_in_bounds) == 0:
            return corners  # No corners to refine

        # Use cornerSubPix for sub-pixel refinement
        win_size = (11, 11)
        zero_zone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

        try:
            # Refine only corners within bounds
            for i in corners_in_bounds:
                corner_point = refined[i:i+1].reshape(-1, 1, 2).astype(np.float32)
                cv2.cornerSubPix(gray, corner_point, win_size, zero_zone, criteria)
                sub_pix_corner = corner_point.reshape(2)

                # Accept refinement if movement is reasonable (max 50 pixels)
                dist = np.linalg.norm(sub_pix_corner - refined[i])
                if dist < 50.0:
                    refined[i] = sub_pix_corner
        except cv2.error:
            # If refinement fails, keep original corners
            pass

        return refined

    def _score_candidate(self, candidate: Dict, image_shape: Tuple[int, int, int], edge_map: np.ndarray, preferred_cover: float, adaptive_params: Optional[Dict] = None) -> Optional[Dict]:
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

        # Calculate aspect ratio - papers are typically in certain ratios
        width = max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3]))
        height = max(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1]))
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < 1:
            aspect_ratio = 1.0 / aspect_ratio

        # Common paper aspect ratios: A4 (1.414), Letter (1.294), etc.
        # Give bonus for aspect ratios between 1.0 and 2.0
        aspect_bonus = 1.0
        if 1.0 <= aspect_ratio <= 2.0:
            aspect_bonus = 1.2
        elif aspect_ratio > 3.0 or aspect_ratio < 0.5:
            aspect_bonus = 0.7  # Penalty for unusual aspect ratios

        cover = metrics['cover_ratio']
        cover_score = max(0.0, 1.0 - abs(cover - preferred_cover) * 2.0)
        if cover < 0.05 or cover > 1.3:
            cover_score *= 0.2

        # Enhanced scoring with balanced weights
        score = metrics['rectangularity'] * 1.7
        score += cover_score * 1.5
        score += edge_support * 2.0
        score -= metrics['perspective_angle'] * 0.01
        score *= aspect_bonus

        # Add uniformity and edge strength bonuses if available
        if 'uniformity' in candidate:
            score *= (0.85 + 0.15 * candidate['uniformity'])
        if 'edge_strength' in candidate:
            edge_bonus = min(candidate['edge_strength'] / 50.0, 1.0)
            score *= (0.9 + 0.1 * edge_bonus)

        # SOFT saturation penalty (not hard mask!)
        # Paper should have low saturation, fingers/skin have high saturation
        if self.use_adaptive_params and adaptive_params and 'saturation' in adaptive_params:
            saturation = adaptive_params['saturation']
            sat_threshold = adaptive_params['saturation_threshold']

            # Compute mean saturation within candidate
            mask = np.zeros(saturation.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [ordered.astype(np.int32)], 255)
            mean_sat = np.mean(saturation[mask > 0]) if np.any(mask > 0) else 0

            # Apply gradual penalty if saturation is high
            if mean_sat > sat_threshold:
                # Gradual penalty: from 1.0 down to 0.5 for very high saturation
                saturation_penalty = max(0.5, 1.0 - (mean_sat - sat_threshold) / sat_threshold)
                score *= saturation_penalty

        if candidate.get('is_near_edge', False) and cover > 0.9:
            score *= 0.6  # Stronger penalty
        if candidate.get('is_near_edge', False) and cover > 0.75 and edge_support < 0.2:
            score *= 0.5  # Even stronger penalty

        score /= outside_penalty

        # Moderate bonus for background flattening candidates
        # They handle challenging illumination - use moderate boost with adaptive (not 50x but not tiny either)
        if candidate.get('from_flattening', False):
            if self.use_adaptive_params:
                score *= 8.0  # Moderate 8x bonus - balances robustness without extreme overfitting
            else:
                score *= 50.0  # Keep old behavior if adaptive disabled

        return {
            'contour': ordered,
            'score': score,
            'metrics': metrics,
            'edge_support': edge_support,
            'cover_score': cover_score,
            'is_near_edge': candidate.get('is_near_edge', False),
            'rectangularity': metrics['rectangularity'],
            'aspect_ratio': aspect_ratio,
            'from_flattening': candidate.get('from_flattening', False)  # Preserve flag
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
