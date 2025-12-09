#!/usr/bin/env python3
"""
Visual corner editor for ground truth data
Controls:
- Click and drag corners to adjust position
- '0-3' - Select corner by number
- Arrow keys - Move selected corner (1px, Shift=10px)
- 'n' - Next image
- 'p' - Previous image
- 's' - Save current corners
- 'r' - Reset to original corners
- 'a' - Auto-detect corners
- 'm' - Mark/unmark image for review
- 'q' or ESC - Quit
"""

import cv2
import json
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paper_detection import PaperDetector

# Paths (relative to this file in paper_detection/tests/)
TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR.parent / "data"
TEST_IMAGES_DIR = DATA_DIR / "images"
GROUND_TRUTH_JSON = DATA_DIR / "corners.json"

# UI constants
CORNER_RADIUS = 15
CORNER_COLOR_IDLE = (0, 255, 0)  # Green
CORNER_COLOR_HOVER = (0, 255, 255)  # Yellow
CORNER_COLOR_DRAG = (0, 0, 255)  # Red
CORNER_COLOR_SELECTED = (255, 0, 255)  # Magenta
LINE_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Zoom window constants
ZOOM_SIZE = 200  # Size of zoom window
ZOOM_FACTOR = 4  # Magnification factor
ZOOM_REGION = 50  # Size of region to capture around corner

# Display window constants
MAX_DISPLAY_WIDTH = 1400  # Maximum width of display window
MAX_DISPLAY_HEIGHT = 1000  # Maximum height of display window

class CornerEditor:
    def __init__(self):
        self.load_ground_truth()
        self.load_image_list()
        self.current_index = 0
        self.dragging_corner = None
        self.hover_corner = None
        self.selected_corner = None  # For keyboard navigation
        self.corners_modified = False
        self.original_corners = None
        self.detector = PaperDetector()
        self.marked_images = set()  # Set of marked image names
        self.display_scale = 1.0  # Scale factor for display
        self.display_offset = (0, 0)  # Offset for centering image

        # Window name
        self.window_name = "Corner Editor"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def load_ground_truth(self):
        """Load ground truth JSON"""
        with open(GROUND_TRUTH_JSON, 'r') as f:
            self.ground_truth = json.load(f)

    def save_ground_truth(self):
        """Save ground truth JSON"""
        with open(GROUND_TRUTH_JSON, 'w') as f:
            json.dump(self.ground_truth, f, indent=4)
        print(f"✓ Saved: {self.current_image_name}")

    def load_image_list(self):
        """Load list of images from ground truth that exist in test_images"""
        all_images = sorted(self.ground_truth.keys())

        # Filter to only images that exist
        self.image_names = []
        for img_name in all_images:
            img_path = TEST_IMAGES_DIR / img_name
            if img_path.exists():
                self.image_names.append(img_name)
            else:
                print(f"⚠ Skipping missing image: {img_name}")

        print(f"Loaded {len(self.image_names)} images (out of {len(all_images)} in JSON)")

    def load_current_image(self):
        """Load current image and its corners"""
        self.current_image_name = self.image_names[self.current_index]
        image_path = TEST_IMAGES_DIR / self.current_image_name

        self.image = cv2.imread(str(image_path))
        if self.image is None:
            print(f"✗ Failed to load: {self.current_image_name}")
            return False

        self.height, self.width = self.image.shape[:2]

        # Calculate display scale to fit in window
        scale_w = MAX_DISPLAY_WIDTH / self.width
        scale_h = MAX_DISPLAY_HEIGHT / self.height
        self.display_scale = min(scale_w, scale_h, 1.0)  # Don't scale up, only down

        # Calculate display size and centering offset
        self.display_width = int(self.width * self.display_scale)
        self.display_height = int(self.height * self.display_scale)
        self.display_offset = (
            (MAX_DISPLAY_WIDTH - self.display_width) // 2,
            (MAX_DISPLAY_HEIGHT - self.display_height) // 2
        )

        # Load corners from ground truth
        gt_data = self.ground_truth[self.current_image_name]
        relative_corners = np.array(gt_data["corners"], dtype=np.float32)

        # Convert to absolute pixels
        self.corners = relative_corners.copy()
        self.corners[:, 0] *= self.width
        self.corners[:, 1] *= self.height

        # Store original corners
        self.original_corners = self.corners.copy()
        self.corners_modified = False

        return True

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        # Convert display coordinates to image coordinates
        x_img = (x - self.display_offset[0]) / self.display_scale
        y_img = (y - self.display_offset[1]) / self.display_scale
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on a corner
            for i, corner in enumerate(self.corners):
                dist = np.linalg.norm([x_img - corner[0], y_img - corner[1]])
                if dist < CORNER_RADIUS * 2:
                    self.dragging_corner = i
                    self.selected_corner = i  # Also select it for keyboard control
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_corner is not None:
                # Drag corner
                self.corners[self.dragging_corner] = [x_img, y_img]
                self.corners_modified = True
            else:
                # Check hover
                self.hover_corner = None
                for i, corner in enumerate(self.corners):
                    dist = np.linalg.norm([x_img - corner[0], y_img - corner[1]])
                    if dist < CORNER_RADIUS * 2:
                        self.hover_corner = i
                        break

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_corner = None

    def draw_zoom_window(self, display, corner_pos_display, corner_index):
        """Draw a magnified view of the area around the corner"""
        # Convert display position back to image coordinates for sampling
        corner_img = self.corners[corner_index]
        cx, cy = int(corner_img[0]), int(corner_img[1])

        # Calculate region to capture from original image
        x1 = max(0, cx - ZOOM_REGION)
        y1 = max(0, cy - ZOOM_REGION)
        x2 = min(self.width, cx + ZOOM_REGION)
        y2 = min(self.height, cy + ZOOM_REGION)

        # Extract region
        region = self.image[y1:y2, x1:x2].copy()

        if region.size == 0:
            return

        # Resize to zoom size
        zoomed = cv2.resize(region, (ZOOM_SIZE, ZOOM_SIZE), interpolation=cv2.INTER_LINEAR)

        # Calculate center position in zoomed image
        center_x = int((cx - x1) / (x2 - x1) * ZOOM_SIZE)
        center_y = int((cy - y1) / (y2 - y1) * ZOOM_SIZE)

        # Draw crosshair at center
        cv2.line(zoomed, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
        cv2.line(zoomed, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)
        cv2.circle(zoomed, (center_x, center_y), 5, (0, 0, 255), -1)

        # Add border
        cv2.rectangle(zoomed, (0, 0), (ZOOM_SIZE - 1, ZOOM_SIZE - 1), (255, 255, 255), 3)
        cv2.rectangle(zoomed, (0, 0), (ZOOM_SIZE - 1, ZOOM_SIZE - 1), (0, 0, 0), 1)

        # Add label
        label = f"Corner {corner_index}"
        cv2.putText(zoomed, label, (10, 25), FONT, 0.6, (0, 0, 0), 3)
        cv2.putText(zoomed, label, (10, 25), FONT, 0.6, (255, 255, 255), 2)

        # Position zoom window near the corner (using display coordinates)
        # Offset to the right and down to avoid obscuring the corner
        cx_display, cy_display = int(corner_pos_display[0]), int(corner_pos_display[1])
        offset_x = 40
        offset_y = 40

        zoom_x = cx_display + offset_x
        zoom_y = cy_display + offset_y

        # If too close to right edge, show on left side
        if zoom_x + ZOOM_SIZE > self.display_width - 10:
            zoom_x = cx_display - ZOOM_SIZE - offset_x

        # If too close to bottom edge, show on top
        if zoom_y + ZOOM_SIZE > self.display_height - 10:
            zoom_y = cy_display - ZOOM_SIZE - offset_y

        # Make sure it's not off screen
        zoom_x = max(10, min(zoom_x, self.display_width - ZOOM_SIZE - 10))
        zoom_y = max(10, min(zoom_y, self.display_height - ZOOM_SIZE - 10))

        # Overlay on display
        display[zoom_y:zoom_y + ZOOM_SIZE, zoom_x:zoom_x + ZOOM_SIZE] = zoomed

    def draw_ui(self):
        """Draw the UI"""
        # Resize image for display
        if self.display_scale != 1.0:
            display = cv2.resize(self.image, (self.display_width, self.display_height))
        else:
            display = self.image.copy()
        
        # Scale corners for display
        display_corners = self.corners * self.display_scale

        # Draw lines between corners
        for i in range(4):
            p1 = tuple(display_corners[i].astype(int))
            p2 = tuple(display_corners[(i + 1) % 4].astype(int))
            cv2.line(display, p1, p2, CORNER_COLOR_IDLE, LINE_THICKNESS)

        # Draw corners
        for i, corner in enumerate(display_corners):
            pos = tuple(corner.astype(int))

            # Choose color based on state
            if self.dragging_corner == i:
                color = CORNER_COLOR_DRAG
                radius = CORNER_RADIUS + 5
            elif self.selected_corner == i:
                color = CORNER_COLOR_SELECTED
                radius = CORNER_RADIUS + 4
            elif self.hover_corner == i:
                color = CORNER_COLOR_HOVER
                radius = CORNER_RADIUS + 3
            else:
                color = CORNER_COLOR_IDLE
                radius = CORNER_RADIUS

            cv2.circle(display, pos, radius, color, -1)
            cv2.circle(display, pos, radius + 2, (255, 255, 255), 2)

            # Draw corner number
            text_pos = (pos[0] + 20, pos[1])
            cv2.putText(display, f"{i}", text_pos, FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS + 1)
            cv2.putText(display, f"{i}", text_pos, FONT, FONT_SCALE, color, FONT_THICKNESS)

        # Draw info text
        info_y = 30
        is_marked = self.current_image_name in self.marked_images
        info_texts = [
            f"Image: {self.current_index + 1}/{len(self.image_names)} - {self.current_image_name}",
            f"Modified: {'YES' if self.corners_modified else 'NO'} | Marked: {'YES ★' if is_marked else 'NO'}",
            f"Selected corner: {self.selected_corner if self.selected_corner is not None else 'None'}",
            "",
            "Controls:",
            "  Click & Drag - Move corner",
            "  0-3 - Select corner",
            "  Arrow keys - Move corner (Shift=10px)",
            "  M - Mark/unmark image",
            "  N - Next | P - Previous",
            "  S - Save | R - Reset | A - Auto",
            "  Q/ESC - Quit",
        ]

        for i, text in enumerate(info_texts):
            y_pos = info_y + i * 25
            # Draw text background
            (text_width, text_height), _ = cv2.getTextSize(text, FONT, 0.6, 2)
            cv2.rectangle(display, (5, y_pos - 20), (15 + text_width, y_pos + 5), (0, 0, 0), -1)
            # Draw text
            cv2.putText(display, text, (10, y_pos), FONT, 0.6, (255, 255, 255), 2)

        # Draw zoom window if dragging a corner
        if self.dragging_corner is not None:
            # Pass original corner position for zoom (not scaled)
            self.draw_zoom_window(display, display_corners[self.dragging_corner], self.dragging_corner)
        
        # Create canvas with consistent size
        canvas = np.zeros((MAX_DISPLAY_HEIGHT, MAX_DISPLAY_WIDTH, 3), dtype=np.uint8)
        # Center the display on canvas
        canvas[self.display_offset[1]:self.display_offset[1] + self.display_height,
               self.display_offset[0]:self.display_offset[0] + self.display_width] = display

        return canvas

    def save_current_corners(self):
        """Save current corners to ground truth"""
        # Convert to relative coordinates
        relative_corners = self.corners.copy()
        relative_corners[:, 0] /= self.width
        relative_corners[:, 1] /= self.height

        # Round to 3 decimal places
        relative_corners = np.round(relative_corners, 3)

        # Update ground truth
        self.ground_truth[self.current_image_name]["corners"] = relative_corners.tolist()

        # Update note
        if "note" in self.ground_truth[self.current_image_name]:
            if "TEMPORARY" in self.ground_truth[self.current_image_name]["note"]:
                self.ground_truth[self.current_image_name]["note"] = "Manually adjusted"
        else:
            self.ground_truth[self.current_image_name]["note"] = "Manually adjusted"

        self.save_ground_truth()
        self.corners_modified = False

    def reset_corners(self):
        """Reset to original corners"""
        self.corners = self.original_corners.copy()
        self.corners_modified = False
        print(f"↻ Reset: {self.current_image_name}")

    def auto_detect_corners(self):
        """Auto-detect corners using PaperDetector"""
        detected = self.detector.detect(self.image)
        if detected is not None:
            self.corners = detected.copy()
            self.corners_modified = True
            print(f"✓ Auto-detected: {self.current_image_name}")
        else:
            print(f"✗ Detection failed: {self.current_image_name}")

    def move_selected_corner(self, dx, dy):
        """Move the selected corner by dx, dy pixels"""
        if self.selected_corner is not None:
            self.corners[self.selected_corner][0] += dx
            self.corners[self.selected_corner][1] += dy
            self.corners_modified = True

    def toggle_mark(self):
        """Toggle mark on current image"""
        if self.current_image_name in self.marked_images:
            self.marked_images.remove(self.current_image_name)
            print(f"☐ Unmarked: {self.current_image_name}")
        else:
            self.marked_images.add(self.current_image_name)
            print(f"★ Marked: {self.current_image_name}")

    def next_image(self):
        """Go to next image"""
        if self.corners_modified:
            print("💾 Auto-saving changes before moving to next image...")
            self.save_current_corners()

        self.current_index = (self.current_index + 1) % len(self.image_names)
        self.load_current_image()

    def prev_image(self):
        """Go to previous image"""
        if self.corners_modified:
            print("💾 Auto-saving changes before moving to previous image...")
            self.save_current_corners()

        self.current_index = (self.current_index - 1) % len(self.image_names)
        self.load_current_image()

    def run(self):
        """Main loop"""
        if not self.load_current_image():
            return

        print("\n" + "="*60)
        print("Corner Editor Started")
        print("="*60)
        print("\nControls:")
        print("  Click & Drag - Move corner")
        print("  0-3 - Select corner by number")
        print("  Arrow keys - Move selected corner (1px, Shift=10px)")
        print("  M - Mark/unmark image for review")
        print("  N - Next image")
        print("  P - Previous image")
        print("  S - Save corners")
        print("  R - Reset to original")
        print("  A - Auto-detect")
        print("  Q/ESC - Quit")
        print("="*60 + "\n")

        while True:
            # Draw UI
            display = self.draw_ui()
            cv2.imshow(self.window_name, display)

            # Handle keyboard
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                if self.corners_modified:
                    print("💾 Auto-saving changes before exiting...")
                    self.save_current_corners()
                break
            elif key == ord('n'):  # Next
                self.next_image()
            elif key == ord('p'):  # Previous
                self.prev_image()
            elif key == ord('s'):  # Save
                self.save_current_corners()
            elif key == ord('r'):  # Reset
                self.reset_corners()
            elif key == ord('a'):  # Auto-detect
                self.auto_detect_corners()
            elif key == ord('m'):  # Mark/unmark
                self.toggle_mark()
            # Select corner by number (0-3)
            elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
                self.selected_corner = int(chr(key))
                print(f"Selected corner: {self.selected_corner}")
            # Arrow keys for moving selected corner
            elif key == 81 or key == 2:  # Left arrow
                shift_pressed = cv2.waitKey(1) & 0xFF == 225  # Check for shift
                self.move_selected_corner(-10 if shift_pressed else -1, 0)
            elif key == 83 or key == 3:  # Right arrow
                shift_pressed = cv2.waitKey(1) & 0xFF == 225
                self.move_selected_corner(10 if shift_pressed else 1, 0)
            elif key == 82 or key == 0:  # Up arrow
                shift_pressed = cv2.waitKey(1) & 0xFF == 225
                self.move_selected_corner(0, -10 if shift_pressed else -1)
            elif key == 84 or key == 1:  # Down arrow
                shift_pressed = cv2.waitKey(1) & 0xFF == 225
                self.move_selected_corner(0, 10 if shift_pressed else 1)

        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("Editor closed")
        print("="*60)
        
        # Print marked images
        if self.marked_images:
            print("\n" + "="*60)
            print(f"Marked images ({len(self.marked_images)}):")
            print("="*60)
            for img_name in sorted(self.marked_images):
                print(f"  ★ {img_name}")
            print("="*60)
        else:
            print("\nNo images were marked.")

if __name__ == "__main__":
    editor = CornerEditor()
    editor.run()
