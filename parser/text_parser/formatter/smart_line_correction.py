"""
Smart line correction module

Applies intelligent corrections to detected lines before formatting.
"""
from typing import List
import numpy as np
import cv2
from pathlib import Path
from parser.ai import send_prompt_with_schema
from .line import Line
import uuid
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Counter for unique line image filenames (thread-safe)
_line_counter = 0
_counter_lock = threading.Lock()


def smart_line_correction(line: Line, image: np.ndarray, debug: bool = False) -> Line:
    """
    Apply smart corrections to a single line

    Args:
        line: Line object to correct (with words in local cropped image coordinates)
        image: Cropped line image (BGR format)
        debug: If True, keep temporary line images for debugging

    Returns:
        Corrected Line object (with words in local cropped image coordinates)
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

    # Get image dimensions for relative coordinates
    img_height, img_width = image.shape[:2]

    # Convert current OCR bounding boxes to relative coordinates (already in local space)
    ocr_tokens = []
    for word in line.words:
        ocr_tokens.append({
            "text": word.text,
            "x": round(word.bounds.left / img_width, 3),
            "y": round(word.bounds.top / img_height, 3),
            "w": round(word.bounds.width / img_width, 3),
            "h": round(word.bounds.height / img_height, 3),
            "confidence": round(word.confidence, 2)
        })

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
        import json
        ocr_tokens_str = json.dumps(ocr_tokens, ensure_ascii=False)

        system_prompt = (
            "You are a smart OCR corrector for song sheets.\n"
            "Read the line from the IMAGE and output the TRUE tokens with approximate bounding boxes.\n"
            "Each line is either a CHORD LINE or a TEXT LINE (never mixed).\n\n"

            "RETURN FORMAT:\n"
            "{ \"isChordLine\": <bool>, \"tokens\": [ {\"text\": str, \"x\": num, \"y\": num, \"w\": num, \"h\": num} ] }\n\n"

            "BOUNDING BOX FORMAT (IMPORTANT):\n"
            "- ALL coordinates must be RELATIVE (0.0 to 1.0), not pixels!\n"
            "- x: left position (0.0 = left edge, 1.0 = right edge)\n"
            "- y: top position (0.0 = top edge, 1.0 = bottom edge)\n"
            "- w: width (0.0 to 1.0 of image width)\n"
            "- h: height (0.0 to 1.0 of image height)\n"
            "- Example: {\"text\": \"Hello\", \"x\": 0.1, \"y\": 0.3, \"w\": 0.2, \"h\": 0.4}\n\n"

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
            "- Root note: A-G (uppercase) or a-g (lowercase).\n"
            "  IMPORTANT: Lowercase a-g are VALID chord roots indicating minor chords!\n"
            "  Examples: 'd' = D minor, 'a' = A minor, 'C' = C major, 'G' = G major\n"
            "- Optionally followed by # or b (sharp/flat).\n"
            "- Optional quality/suffix: m, maj, min, dim, aug, sus, add.\n"
            "- Optional extensions: digits like 2,4,5,6,7,9,11,13.\n"
            "- Optional slash bass: /A-G or /a-g with optional # or b.\n"
            "- Allowed characters in chord token: A-G a-g 0-9 # b / +.\n"
            "- Any other character (quotes, commas, dots, colons, weird symbols) makes it INVALID unless removed.\n\n"

            "CHORD LINE RULES (isChordLine=true) — CRITICAL:\n"
            "1) EACH returned token MUST be ONE valid chord (per definition above).\n"
            "2) A token MUST NOT contain two roots combined (examples of INVALID: 'EC', 'CA', 'GD', 'E C', 'C/A G').\n"
            "3) If you see two chords close together, you MUST output TWO tokens with TWO separate bboxes.\n"
            "4) If you cannot confidently split them, OMIT them (better empty than wrong).\n"
            "5) Before outputting, CHECK validity. If invalid, either split into multiple valid chords or omit.\n"
            "6) CONTEXT CLUE: If a line contains ONLY single letters (a-g, A-G) with optional modifiers (#/b/m/7/etc),\n"
            "   it is HIGHLY LIKELY a chord line, not text. Example: 'd C a d' = valid chord progression.\n\n"

            "TEXT LINE RULES (isChordLine=false):\n"
            "- Output only normal words (multiple characters forming meaningful words).\n"
            "- Isolated single letters (a-g, A-G) are almost always CHORDS, not text words.\n\n"

            "Return ONLY valid JSON. No extra text."
        )

        user_prompt = (
            f"OCR DETECTED TOKENS (with relative coordinates and confidence):\n"
            f"{ocr_tokens_str}\n\n"

            "NOTE: OCR results may contain errors (typos, split words, wrong spacing, low confidence).\n"
            "Your job is to read the ACTUAL TEXT from the IMAGE and return corrected tokens.\n"
            "Use OCR results as a HINT for approximate positions, but trust the IMAGE for the actual text."
        )

        result = send_prompt_with_schema(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=schema,
            image_path=str(output_path),
            schema_name="line_correction"
        ) or {}
        tokens = result.get("tokens", [])

        if tokens:
            # Import ReadWordData and Bounds
            from ..ocr.read_word_data import ReadWordData
            from parser.bounds import Bounds

            # Replace line words with AI-detected tokens (in local coordinates)
            line.words = []
            for token in tokens:
                # Convert relative coordinates (0-1) to absolute pixels in local cropped image
                bounds = Bounds(
                    left=token.get("x", 0) * img_width,
                    top=token.get("y", 0) * img_height,
                    width=token.get("w", 0) * img_width,
                    height=token.get("h", 0) * img_height
                )
                line.words.append(ReadWordData(token.get("text", ""), bounds, token.get("confidence", 100.0)))

            line.avgConfidence = 100.0
            line.chordLinePossibility = 1.0 if result.get("isChordLine") else 0.0
    except Exception:  # Keep original line on AI failure
        pass  # Silently skip failed corrections
    finally:
        # Clean up temporary image if not in debug mode
        if not debug:
            try:
                import os
                if os.path.exists(str(output_path)):
                    os.remove(str(output_path))
            except Exception as e:
                print(f"Warning: Failed to clean up temp file: {e}")

    # Filter out empty words
    line.words = [word for word in line.words if word.text.strip() != ""]

    return line


def smart_lines_correction(lines: List[Line], image: np.ndarray, max_workers: int = 4, debug: bool = False) -> List[Line]:
    """
    Apply smart corrections to all lines in parallel

    Args:
        lines: List of Line objects to correct
        image_bgr: Original image (BGR format) from which text was extracted
        max_workers: Maximum number of parallel workers (default: 4)
        debug: If True, keep temporary line images for debugging

    Returns:
        List of corrected Line objects
    """
    def process_line(index: int, line: Line) -> tuple[int, Line]:
        """Process a single line and return its index and result"""
        should_correct = line.avgConfidence < 94 or line.chordLinePossibility > 0.4
        if should_correct:
            # Crop image to line bounds with padding
            pad = 5
            top = max(0, int(line.bounds.top) - pad)
            bottom = min(image.shape[0], int(line.bounds.top + line.bounds.height) + pad)
            left = max(0, int(line.bounds.left) - pad)
            right = min(image.shape[1], int(line.bounds.left + line.bounds.width) + pad)

            croped_image = image[top:bottom, left:right]

            # Convert line words to local coordinates before correction
            from ..ocr.read_word_data import ReadWordData
            from parser.bounds import Bounds
            from copy import deepcopy

            local_line = deepcopy(line)
            for word in local_line.words:
                word.bounds.left -= left
                word.bounds.top -= top

            # Apply AI correction (works with local coordinates)
            corrected_line = smart_line_correction(local_line, croped_image, debug=debug)

            # Convert back to global coordinates
            for word in corrected_line.words:
                word.bounds.left += left
                word.bounds.top += top

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
