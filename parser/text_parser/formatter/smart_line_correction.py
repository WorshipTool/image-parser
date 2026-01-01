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
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Counter for unique line image filenames (thread-safe)
_line_counter = 0
_counter_lock = threading.Lock()

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

    # Thread-safe counter increment
    with _counter_lock:
        counter_value = _line_counter
        _line_counter += 1

    output_path = temp_dir / f"line_{counter_value:04d}_{rand_suffix}.jpg"
    cv2.imwrite(str(output_path), image)

    #TODO: its not necessary to save image to disk, can be sent as 64 string directly
    try:
        schema = {
            "type": "object",
            "properties": {
                "isChordLine": {"type": "boolean"},
                "tokens": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "w": {"type": "number"},
                            "h": {"type": "number"}
                        },
                        "required": ["text", "x", "y", "w", "h"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["isChordLine", "tokens"],
            "additionalProperties": False
        }
        prompt = (
            "Read the line ONLY from the IMAGE and output the TRUE tokens with approximate bounding boxes.\n"
            "This is a CHORD LINE or a TEXT LINE (never mixed).\n\n"

            "Return ONLY valid JSON:\n"
            "{ \"isChordLine\": <bool>, \"tokens\": [ {\"text\": str, \"x\": num, \"y\": num, \"w\": num, \"h\": num} ] }\n\n"

            "GENERAL RULES:\n"
            "- Output tokens in left-to-right order.\n"
            "- Do NOT invent text.\n"
            "- Do NOT output punctuation-only tokens.\n"
            "- If the line is only noise (commas/quotes/strokes), return isChordLine=false and tokens=[].\n\n"

            "LINE TYPE:\n"
            "- isChordLine=true  -> chords only\n"
            "- isChordLine=false -> lyric words only\n\n"

            "VALID CHORD DEFINITION (MUST PASS):\n"
            "- A valid chord token must have EXACTLY ONE root note.\n"
            "- Root note: A, B, C, D, E, F, or G (optionally followed by # or b).\n"
            "- Optional quality/suffix: m, maj, min, dim, aug, sus, add.\n"
            "- Optional extensions: digits like 2,4,5,6,7,9,11,13.\n"
            "- Optional slash bass: /A.. /G with optional # or b.\n"
            "- Allowed characters in chord token: A-G a-g 0-9 # b / +.\n"
            "- Any other character (quotes, commas, dots, colons, weird symbols) makes it INVALID unless removed.\n\n"

            "CHORD LINE RULES (isChordLine=true) — CRITICAL:\n"
            "1) EACH returned token MUST be ONE valid chord (per definition above).\n"
            "2) A token MUST NOT contain two roots combined (examples of INVALID: 'EC', 'CA', 'GD', 'E C', 'C/A G').\n"
            "3) If you see two chords close together, you MUST output TWO tokens with TWO separate bboxes.\n"
            "4) If you cannot confidently split them, OMIT them (better empty than wrong).\n"
            "5) Before outputting, CHECK validity. If invalid, either split into multiple valid chords or omit.\n\n"

            "TEXT LINE RULES (isChordLine=false):\n"
            "- Output only normal words.\n"
            "- Do NOT output chord-like tokens.\n\n"

            "Return ONLY JSON. No extra text."
        )

        result = send_image_and_question(str(output_path), prompt, json_schema=schema) or {}
        tokens = result.get("tokens", [])

        if tokens:
            # Import ReadWordData and Bounds
            from ..ocr.read_word_data import ReadWordData, Bounds

            # Replace line words with AI-detected tokens
            line.words = []
            for token in tokens:
                bounds = Bounds(
                    left=float(token.get("x", 0)),
                    top=float(token.get("y", 0)),
                    width=float(token.get("w", 0)),
                    height=float(token.get("h", 0))
                )
                word = ReadWordData(
                    text=token.get("text", ""),
                    bounds=bounds,
                    confidence=100.0
                )
                line.words.append(word)

            line.avgConfidence = 100.0
            line.chordLinePossibility = 1.0 if result.get("isChordLine") else 0.0
    except Exception as exc:  # Keep original line on AI failure
        pass  # Silently skip failed corrections

    
    # Filter out empty words
    line.words = [word for word in line.words if word.text.strip() != ""]

    return line


def smart_lines_correction(lines: List[Line], image: np.ndarray, max_workers: int = 4) -> List[Line]:
    """
    Apply smart corrections to all lines in parallel

    Args:
        lines: List of Line objects to correct
        image_bgr: Original image (BGR format) from which text was extracted
        max_workers: Maximum number of parallel workers (default: 4)

    Returns:
        List of corrected Line objects
    """
    def process_line(index: int, line: Line) -> tuple[int, Line]:
        """Process a single line and return its index and result"""
        should_correct = line.avgConfidence < 94 or line.chordLinePossibility > 0.5
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
            return (index, corrected_line, True)
        else:
            return (index, line, False)

    # Prepare results dictionary to preserve order
    results = {}

    # Create progress bar
    with tqdm(total=len(lines), desc="Processing lines", unit="line", ncols=100) as pbar:
        # Process lines in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_line, i, line): i for i, line in enumerate(lines)}

            # Process completed tasks as they finish
            for future in as_completed(futures):
                index, corrected_line, was_corrected = future.result()
                results[index] = corrected_line

                # Update progress bar
                if was_corrected:
                    pbar.set_postfix_str(f"AI corrected")
                else:
                    pbar.set_postfix_str(f"Skipped (high conf)")
                pbar.update(1)

    # Return results in original order
    corrected_lines = [results[i] for i in range(len(lines))]
    return corrected_lines
