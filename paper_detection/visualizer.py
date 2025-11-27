"""
Visualization of detected paper
"""

import cv2
import numpy as np
from typing import Tuple


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

        # Blue frame
        if draw_border:
            cv2.polylines(
                result,
                [corners_int],
                isClosed=True,
                color=self.border_color,
                thickness=self.border_thickness
            )

            # Draw circles at corners
            for corner in corners_int:
                cv2.circle(
                    result,
                    tuple(corner),
                    radius=5,
                    color=self.border_color,
                    thickness=-1
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
