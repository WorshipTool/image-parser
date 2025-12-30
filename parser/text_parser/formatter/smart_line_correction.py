"""
Smart line correction module

Applies intelligent corrections to detected lines before formatting.
"""
from typing import List
import json
import numpy as np
import cv2
from pathlib import Path
from ai import send_image_and_question
from .line import Line
import uuid

# Counter for unique line image filenames
_line_counter = 0

def smart_line_correction(line: Line, image: np.ndarray) -> Line:
    """
    Apply smart corrections to a single line

    Args:
        line: Line object to correct
        image: Cropped line image (BGR format)

    Returns:
        Corrected Line object
    """
    global _line_counter

    # Save line image to temp folder
    current_file = Path(__file__).resolve()
    image_parser_root = current_file.parent.parent.parent.parent
    temp_dir = image_parser_root / "temp" / "line_corrections"
    temp_dir.mkdir(parents=True, exist_ok=True)
    rand_suffix = uuid.uuid4().hex[:8]
    output_path = temp_dir / f"line_{_line_counter:04d}_{rand_suffix}.jpg"
    cv2.imwrite(str(output_path), image)

    #TODO: its not necessary to save image to disk, can be sent as 64 string directly


    _line_counter += 1
    try:
        schema = {
            "type": "object",
            "properties": {
                "words": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "isChordLine": {"type": "boolean"}
            },
            "required": ["words", "isChordLine"],
            "additionalProperties": False
        }
        prompt = (
            "IMPORTANT: The image is the ONLY source of truth. The OCR tokens are often wrong.\n"
            "Do NOT blindly copy the input. Correct tokens based on what you SEE in the image.\n\n"

            "First, decide the line type and set `isChordLine`:\n"
            "- isChordLine = true if the line contains ONLY chord symbols "
            "(e.g. C, Dm7, G#dim, C/E, D7add9/F#).\n"
            "- isChordLine = false if the line contains normal lyric text (words).\n\n"

            "Then return a JSON object with:\n"
            "- `isChordLine`\n"
            "- `words`: an array with the SAME number of items as the input.\n\n"

            "Rules for `words`:\n"
            "- Each item corresponds to the same position as in the input list.\n"
            "- If an OCR token does NOT match what is visible in the image at that position, correct it.\n"
            "- If there is NO meaningful token visible at that position (OCR noise, random strokes), "
            "return an empty string \"\".\n"
            "- Preserve order and list length at all times.\n\n"

            "Rules depending on `isChordLine`:\n"
            "- If isChordLine = true:\n"
            "  - Every non-empty item MUST be a valid chord symbol.\n"
            "  - Any token that is not a chord MUST be returned as \"\".\n"
            "- If isChordLine = false:\n"
            "  - Tokens that look like chords MUST be returned as \"\".\n"
            "  - Return normal words only.\n\n"

            "Do NOT add, remove, split, or merge tokens.\n"
            "Do NOT invent text or chords.\n"
            "Return ONLY valid JSON that matches the required schema.\n\n"

            "Input OCR tokens:\n"
            f"{json.dumps([w.text for w in line.words], ensure_ascii=False)}"
        )
        result = send_image_and_question(str(output_path), prompt, json_schema=schema) or {}
        tokens = result.get("words", [])

        if tokens:
            min_len = min(len(tokens), len(line.words))
            for word, token in zip(line.words, tokens[:min_len]):
                word.text = token

            if len(tokens) > len(line.words):
                line.words[-1].text += " " + " ".join(tokens[len(line.words):])

            line.avgConfidence = 100.0
            print(f"AI corrected line {_line_counter - 1}: {' '.join(tokens)} | isChordLine: {result.get('isChordLine')}")
            print(output_path, line)
        else:
            print(f"AI line correction failed for line {_line_counter - 1}, keeping original.")
    except Exception as exc:  # Keep original line on AI failure
        print(f"AI line correction skipped: {exc}")



    return line


def smart_lines_correction(lines: List[Line], image: np.ndarray) -> List[Line]:
    """
    Apply smart corrections to all lines

    Args:
        lines: List of Line objects to correct
        image_bgr: Original image (BGR format) from which text was extracted

    Returns:
        List of corrected Line objects
    """
    corrected_lines = []

    for line in lines:
        should_correct = line.avgConfidence < 94
        if should_correct:
            # Crop image to line bounds

            # Add light padding to line bounds
            pad = 5
            top = max(0, int(line.bounds.top) - pad)
            bottom = min(image.shape[0], int(line.bounds.top + line.bounds.height) + pad)
            left = max(0, int(line.bounds.left) - pad)
            right = min(image.shape[1], int(line.bounds.left + line.bounds.width) + pad)

            croped_image = image[top:bottom, left:right]
            corrected_line = smart_line_correction(line, croped_image) 
        else:
            corrected_line = line
        corrected_lines.append(corrected_line)

    return corrected_lines
