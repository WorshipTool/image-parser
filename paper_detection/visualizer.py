"""
Visualization of detected paper
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


class PaperVisualizer:
    """
    Class for visualizing detected paper.

    Allows drawing a blue frame and transparent background
    around the detected paper.
    """

    def __init__(
        self,
        border_color: Tuple[int, int, int] = (255, 100, 0),  # Blue in BGR
        border_thickness: int = 3,
        overlay_color: Tuple[int, int, int] = (255, 200, 100),  # Light blue in BGR
        overlay_alpha: float = 0.3
    ):
        """
        Initialize the visualizer.

        Args:
            border_color: Frame color in BGR format
            border_thickness: Frame thickness in pixels
            overlay_color: Transparent overlay color in BGR format
            overlay_alpha: Overlay transparency (0.0 = transparent, 1.0 = opaque)
        """
        self.border_color = border_color
        self.border_thickness = border_thickness
        self.overlay_color = overlay_color
        self.overlay_alpha = overlay_alpha

    def _clip_line_to_image(self, p1: np.ndarray, p2: np.ndarray, img_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Clip a line segment to image boundaries using Cohen-Sutherland algorithm.

        Args:
            p1: First point [x, y]
            p2: Second point [x, y]
            img_shape: Image shape (height, width)

        Returns:
            Tuple of clipped points (p1_clipped, p2_clipped) or (None, None) if line is completely outside
        """
        h, w = img_shape
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])

        # Cohen-Sutherland clipping
        INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

        def compute_outcode(x, y):
            code = INSIDE
            if x < 0: code |= LEFT
            elif x > w: code |= RIGHT
            if y < 0: code |= TOP
            elif y > h: code |= BOTTOM
            return code

        outcode1 = compute_outcode(x1, y1)
        outcode2 = compute_outcode(x2, y2)

        while True:
            # Both points inside
            if outcode1 == 0 and outcode2 == 0:
                return np.array([x1, y1]), np.array([x2, y2])

            # Both points in same outside region
            if outcode1 & outcode2:
                return None, None

            # At least one point outside, pick it
            outcode_out = outcode1 if outcode1 else outcode2

            # Find intersection point
            if outcode_out & TOP:
                x = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
                y = 0
            elif outcode_out & BOTTOM:
                x = x1 + (x2 - x1) * (h - y1) / (y2 - y1)
                y = h
            elif outcode_out & RIGHT:
                y = y1 + (y2 - y1) * (w - x1) / (x2 - x1)
                x = w
            elif outcode_out & LEFT:
                y = y1 + (y2 - y1) * (0 - x1) / (x2 - x1)
                x = 0

            # Replace point outside with intersection
            if outcode_out == outcode1:
                x1, y1 = x, y
                outcode1 = compute_outcode(x1, y1)
            else:
                x2, y2 = x, y
                outcode2 = compute_outcode(x2, y2)

    def _draw_clipped_polygon(self, image: np.ndarray, corners: np.ndarray, color: Tuple[int, int, int], thickness: int):
        """
        Draw polygon with lines clipped to image boundaries.

        Args:
            image: Image to draw on
            corners: Polygon corners
            color: Line color
            thickness: Line thickness
        """
        h, w = image.shape[:2]

        # Draw each edge with clipping
        for i in range(len(corners)):
            p1 = corners[i]
            p2 = corners[(i + 1) % len(corners)]

            p1_clip, p2_clip = self._clip_line_to_image(p1, p2, (h, w))

            if p1_clip is not None and p2_clip is not None:
                cv2.line(
                    image,
                    tuple(p1_clip.astype(np.int32)),
                    tuple(p2_clip.astype(np.int32)),
                    color,
                    thickness
                )

        # Draw circles at corners that are inside image
        for corner in corners:
            if 0 <= corner[0] <= w and 0 <= corner[1] <= h:
                cv2.circle(
                    image,
                    tuple(corner.astype(np.int32)),
                    radius=5,
                    color=color,
                    thickness=-1
                )

    def visualize(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        draw_border: bool = True,
        draw_overlay: bool = True
    ) -> np.ndarray:
        """
        Visualize detected paper on the image.

        Args:
            image: Input image (BGR format)
            corners: Array with 4 paper corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            draw_border: Whether to draw blue frame
            draw_overlay: Whether to draw transparent background

        Returns:
            Image with visualization
        """
        if image is None or corners is None:
            return image

        # Copy image for visualization
        result = image.copy()

        # Convert corners to int32 for drawing
        corners_int = corners.astype(np.int32)

        # Transparent overlay
        if draw_overlay:
            overlay = result.copy()
            cv2.fillPoly(overlay, [corners_int], self.overlay_color)
            result = cv2.addWeighted(
                overlay,
                self.overlay_alpha,
                result,
                1 - self.overlay_alpha,
                0
            )

        # Blue frame - use clipped drawing to handle corners outside image
        if draw_border:
            self._draw_clipped_polygon(
                result,
                corners,
                self.border_color,
                self.border_thickness
            )

        return result

    def visualize_with_info(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        show_dimensions: bool = True
    ) -> np.ndarray:
        """
        Visualize detected paper with additional information.

        Args:
            image: Input image (BGR format)
            corners: Array with 4 paper corners
            show_dimensions: Whether to display paper dimensions

        Returns:
            Image with visualization and information
        """
        result = self.visualize(image, corners)

        if show_dimensions and corners is not None:
            # Calculate dimensions
            width_top = np.linalg.norm(corners[1] - corners[0])
            width_bottom = np.linalg.norm(corners[2] - corners[3])
            width = int(max(width_top, width_bottom))

            height_left = np.linalg.norm(corners[3] - corners[0])
            height_right = np.linalg.norm(corners[2] - corners[1])
            height = int(max(height_left, height_right))

            # Calculate area
            area = cv2.contourArea(corners.astype(np.int32))

            # Text information
            info_text = [
                f"Width: {width}px",
                f"Height: {height}px",
                f"Area: {int(area)}px2"
            ]

            # Position for text (top left corner)
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(
                    result,
                    text,
                    (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    result,
                    text,
                    (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

        return result

    def visualize_with_metrics(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        metrics: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Visualize detected paper with quality metrics.

        Args:
            image: Input image (BGR format)
            corners: Array with 4 paper corners
            metrics: Dictionary with metrics (cover_ratio, rectangularity, angle, perspective_angle)

        Returns:
            Image with visualization and metrics
        """
        result = self.visualize(image, corners)

        if corners is None:
            return result

        # Text information
        info_text = []

        if metrics is not None:
            # Format metrics
            cover_pct = metrics.get('cover_ratio', 0) * 100
            rect_pct = metrics.get('rectangularity', 0) * 100
            angle = metrics.get('angle', 0)
            persp_angle = metrics.get('perspective_angle', 0)

            info_text = [
                f"Cover: {cover_pct:.1f}%",
                f"Rect: {rect_pct:.1f}%",
                f"Angle: {angle:.1f}deg",
                f"Persp: {persp_angle:.1f}deg"
            ]
        else:
            # Fallback to basic dimensions
            width_top = np.linalg.norm(corners[1] - corners[0])
            width_bottom = np.linalg.norm(corners[2] - corners[3])
            width = int(max(width_top, width_bottom))

            height_left = np.linalg.norm(corners[3] - corners[0])
            height_right = np.linalg.norm(corners[2] - corners[1])
            height = int(max(height_left, height_right))

            info_text = [
                f"Width: {width}px",
                f"Height: {height}px"
            ]

        # Position for text (top left corner)
        y_offset = 30
        for i, text in enumerate(info_text):
            # White outline
            cv2.putText(
                result,
                text,
                (10, y_offset + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            # Black text
            cv2.putText(
                result,
                text,
                (10, y_offset + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return result

    def create_side_by_side(
        self,
        original: np.ndarray,
        visualized: np.ndarray
    ) -> np.ndarray:
        """
        Create an image with original and visualized image side by side.

        Args:
            original: Original image
            visualized: Visualized image

        Returns:
            Combined image
        """
        if original is None or visualized is None:
            return original if original is not None else visualized

        # Ensure both images have the same height
        if original.shape[0] != visualized.shape[0]:
            # Resize visualized image
            height = original.shape[0]
            width = int(visualized.shape[1] * height / visualized.shape[0])
            visualized = cv2.resize(visualized, (width, height))

        # Join horizontally
        result = np.hstack([original, visualized])

        return result
