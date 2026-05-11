"""
Utility functions for text parser formatter
"""
from typing import List
import cv2
import numpy as np
from pathlib import Path
import uuid
from PIL import Image, ImageDraw, ImageFont
from .line import Line
from ..ocr.read_word_data import ReadWordData


def visualize_all_corrected_lines(image: np.ndarray, lines: List[Line], output_path: str, raw_ocr_data: List[ReadWordData] = None) -> None:
    """
    Create a visualization of all corrected lines with text on white background,
    with original image on the right side for comparison.

    Args:
        image: Full image (BGR format)
        lines: List of corrected Line objects with words
        output_path: Path where to save the visualization
        raw_ocr_data: Optional raw OCR data to draw bounding boxes on the original image
    """
    # Create white background with same dimensions as original
    height, width = image.shape[:2]

    # Create combined image: text visualization on left, original on right
    combined_width = width * 2
    combined_image = Image.new('RGB', (combined_width, height), color='white')

    # Draw text visualization on left side
    draw = ImageDraw.Draw(combined_image)

    # Use default PIL font (supports Unicode)
    # Try to use a TrueType font with reasonable size, fallback to default
    try:
        font_size = 40
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()

    for line in lines:
        # Determine if it's a chord line
        is_chord_line = line.chordLinePossibility > 0.4

        # Choose color based on whether it's likely a chord
        # Chords = red, text = black
        if is_chord_line:
            text_color = (255, 0, 0)  # Red for chords (RGB)
        else:
            text_color = (0, 0, 0)  # Black for text (RGB)

        # Use line's center Y for all words in this line (aligned horizontally)
        line_y = int(line.centerY)

        for word in line.words:
            # Get X position from word bounds
            x = int(word.bounds.left)

            # Prepare text to display
            text = word.text if word.text else ""

            # Draw text at the same Y position for entire line
            draw.text((x, line_y), text, font=font, fill=text_color)

    # Convert original image from BGR to RGB and paste on right side
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_pil = Image.fromarray(original_rgb)

    # Draw bounding boxes for raw OCR data if provided
    if raw_ocr_data:
        draw_boxes = ImageDraw.Draw(original_pil)

        # Try to use a small font for confidence values
        try:
            confidence_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            confidence_font = ImageFont.load_default()

        for word_data in raw_ocr_data:
            # Get bounding box coordinates
            left = int(word_data.bounds.left)
            top = int(word_data.bounds.top)
            right = int(left + word_data.bounds.width)
            bottom = int(top + word_data.bounds.height)

            # Draw rectangle around the word (green color, 2px width)
            draw_boxes.rectangle(
                [(left, top), (right, bottom)],
                outline=(0, 255, 0),  # Green color in RGB
                width=2
            )

            # Draw confidence value near the top-left corner of the box
            confidence_text = f"{word_data.confidence:.2f}"
            # Position: slightly above the box
            text_position = (left, max(0, top - 18))

            # Draw background rectangle for text readability
            bbox = draw_boxes.textbbox(text_position, confidence_text, font=confidence_font)
            draw_boxes.rectangle(bbox, fill=(0, 0, 0, 128))  # Semi-transparent black background

            # Draw confidence text in white
            draw_boxes.text(
                text_position,
                confidence_text,
                font=confidence_font,
                fill=(255, 255, 255)  # White color
            )

    combined_image.paste(original_pil, (width, 0))

    # Convert combined image to OpenCV format and save
    vis_image = cv2.cvtColor(np.array(combined_image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, vis_image)


def save_lines_visualization(image: np.ndarray, lines: List[Line], raw_ocr_data: List[ReadWordData] = None) -> None:
    """
    Save visualization of all corrected lines to temp folder.

    Args:
        image: Full image (BGR format)
        lines: List of corrected Line objects with words
        raw_ocr_data: Optional raw OCR data to draw bounding boxes on the original image
    """
    try:
        # Get temp directory path
        current_file = Path(__file__).resolve()
        image_parser_root = current_file.parent.parent.parent.parent
        temp_dir = image_parser_root / "temp" / "line_corrections_full"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        vis_filename = f"all_lines_{uuid.uuid4().hex[:8]}.jpg"
        vis_output_path = temp_dir / vis_filename

        # Create and save visualization
        visualize_all_corrected_lines(image, lines, str(vis_output_path), raw_ocr_data)
        print(f"✓ Visualization saved: {vis_output_path}")
    except Exception as e:
        print(f"⚠ Visualization error: {e}")
        import traceback
        traceback.print_exc()
