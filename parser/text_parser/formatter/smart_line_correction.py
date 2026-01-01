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
                "words": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "isChordLine": {"type": "boolean"}
            },
            "required": ["words", "isChordLine"],
            "additionalProperties": False
        }
        tokens_payload = [
            {
                "i": i,
                
                "x": w.bounds.left,
                "y": w.bounds.top,
                "w": w.bounds.width,
                "h": w.bounds.height
            }
            for i, w in enumerate(line.words)
        ]

        prompt = (
            "Read the line ONLY from the IMAGE.\n"
            "Each output slot corresponds to ONE bounding box (bbox).\n"
            "The OCR text is only a hint and may be wrong.\n\n"

            "Return ONLY valid JSON with:\n"
            "- isChordLine (boolean)\n"
            "- words: array of strings, SAME length and order as input tokens\n\n"

            "BBOX RULES (critical):\n"
            "- For each token i, look ONLY inside its bbox region.\n"
            "- If that bbox contains no clear letter/chord (only commas, quotes, strokes), return \"\".\n"
            "- Do NOT move text between boxes.\n\n"

            "Line type:\n"
            "- isChordLine=true → only chord symbols are visible\n"
            "- isChordLine=false → normal lyric words or empty/noise-only line\n\n"

            "If isChordLine=true:\n"
            "- Non-empty words[i] MUST be valid chord symbols (C, Dm7, G#dim, C/E, D7add9/F#).\n"
            "- Remove ALL punctuation/quotes (: , . ; \" “ ” ‘ ’).\n"
            "- If not a valid chord after cleaning → \"\".\n\n"

            "If isChordLine=false:\n"
            "- Non-empty words[i] must be normal words.\n"
            "- Chord-like tokens must be \"\".\n\n"

            "If ALL bboxes contain only noise:\n"
            "- isChordLine=false\n"
            "- words = all \"\"\n\n"

            "Do NOT invent text.\n"
            "Do NOT change order or length.\n\n"

            "Input tokens with bounding boxes:\n"
            f"{json.dumps(tokens_payload, ensure_ascii=False)}"
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
